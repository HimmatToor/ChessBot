"""Export a trained checkpoint to ONNX for the browser (onnxruntime-web).

    python export_onnx.py checkpoints/tiny-cpu/best.pt

Writes ../web/public/models/chessnet.onnx by default and checks that
onnxruntime reproduces PyTorch's outputs on real positions.

Graph interface (batch size is dynamic):
    input   planes  float32 (batch, 18, 8, 8)
    output  policy  float32 (batch, 1858)  move logits (unmasked)
    output  wdl     float32 (batch, 3)     win/draw/loss logits

Weights are stored as float16 by default (half the download) and cast back
to float32 when the model loads, so all maths still runs in float32 and the
model works on every onnxruntime-web backend.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import chess
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

from chessnet.checkpoint import load_checkpoint
from chessnet.encode import ENCODING_VERSION, NUM_MOVES, NUM_PLANES, encode_board
from chessnet.model import ChessNet

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "web" / "public" / "models" / "chessnet.onnx"


def to_onnx(model: ChessNet) -> onnx.ModelProto:
    buffer = io.BytesIO()
    torch.onnx.export(
        model.eval(),
        (torch.zeros(1, NUM_PLANES, 8, 8),),
        buffer,
        input_names=["planes"],
        output_names=["policy", "wdl"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}, "wdl": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,  # folds BatchNorm into the convolutions
    )
    return onnx.load_from_string(buffer.getvalue())


def store_weights_as_fp16(model: onnx.ModelProto, min_elements: int = 1024) -> None:
    """Stores large float32 weights as float16, each followed by a Cast back to float32."""
    graph = model.graph
    converted = []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= min_elements:
            converted.append(init)
    for init in converted:
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_fp16")
        graph.initializer.remove(init)
        graph.initializer.append(half)
        cast = helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast")
        graph.node.insert(0, cast)


def add_metadata(model: onnx.ModelProto, metadata: dict[str, str]) -> None:
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value


def sample_positions(count: int, seed: int = 0) -> np.ndarray:
    """Encoded positions from random games: realistic inputs for the parity check."""
    rng = random.Random(seed)
    planes = []
    while len(planes) < count:
        board = chess.Board()
        while not board.is_game_over() and len(planes) < count:
            planes.append(encode_board(board))
            board.push(rng.choice(list(board.legal_moves)))
    return np.stack(planes).astype(np.float32)


def compare(model: ChessNet, onnx_bytes: bytes, planes: np.ndarray) -> dict[str, float]:
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    onnx_policy, onnx_wdl = session.run(None, {"planes": planes})
    with torch.no_grad():
        policy, wdl = (t.numpy() for t in model(torch.from_numpy(planes)))
    return {
        "max_policy_diff": float(np.abs(onnx_policy - policy).max()),
        "max_wdl_diff": float(np.abs(onnx_wdl - wdl).max()),
        "top1_agreement": float((onnx_policy.argmax(1) == policy.argmax(1)).mean()),
    }


def export(checkpoint_path: Path, out: Path, fp16: bool = True) -> dict[str, float]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    onnx_model = to_onnx(model)
    if fp16:
        store_weights_as_fp16(onnx_model)
    add_metadata(onnx_model, {
        "encoding_version": str(ENCODING_VERSION),
        "num_moves": str(NUM_MOVES),
        "model_config": json.dumps(model.config.to_dict()),
        "training_step": str(checkpoint.get("step", "")),
        "weights": "fp16" if fp16 else "fp32",
    })
    onnx.checker.check_model(onnx_model)

    onnx_bytes = onnx_model.SerializeToString()
    stats = compare(model, onnx_bytes, sample_positions(256))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onnx_bytes)
    stats["size_mb"] = len(onnx_bytes) / 1e6
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fp32", action="store_true", help="keep float32 weights (twice the size)")
    args = parser.parse_args(argv)

    stats = export(args.checkpoint, args.out, fp16=not args.fp32)
    print(f"wrote {args.out} ({stats['size_mb']:.1f} MB)")
    print(f"onnxruntime vs PyTorch on 256 positions: max policy diff {stats['max_policy_diff']:.2e}, "
          f"max wdl diff {stats['max_wdl_diff']:.2e}, same best move {stats['top1_agreement']:.1%}")
    if stats["size_mb"] > 25:
        print("WARNING: over Cloudflare Pages' 25 MB per-file limit")


if __name__ == "__main__":
    main()
