import io

import chess
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from chessnet.encode import NUM_MOVES, encode_board
from chessnet.model import ChessNet, ModelConfig, count_parameters


def start_planes(batch: int) -> torch.Tensor:
    planes = torch.from_numpy(encode_board(chess.Board())).float()
    return planes.expand(batch, -1, -1, -1).clone()


def test_output_shapes():
    model = ChessNet(ModelConfig.tiny()).eval()
    policy, wdl = model(start_planes(4))
    assert policy.shape == (4, NUM_MOVES)
    assert wdl.shape == (4, 3)


def test_full_size_model_is_browser_sized():
    model = ChessNet(ModelConfig())
    params = count_parameters(model)
    # ~7M params => ~14 MB in fp16, under Cloudflare Pages' 25 MB file limit.
    assert 5_000_000 < params < 9_000_000, params
    policy, wdl = model.eval()(start_planes(2))
    assert policy.shape == (2, NUM_MOVES) and wdl.shape == (2, 3)


def test_se_can_be_disabled():
    with_se = count_parameters(ChessNet(ModelConfig.tiny()))
    without = count_parameters(ChessNet(ModelConfig(blocks=2, channels=32, se_ratio=0, head_channels=8, value_hidden=32)))
    assert without < with_se


def test_tiny_model_can_overfit_a_small_batch():
    """Gradients reach every part of the network: it can memorise 32 random targets."""
    torch.manual_seed(0)
    model = ChessNet(ModelConfig.tiny())
    planes = torch.randint(0, 2, (32, 18, 8, 8)).float()
    move_targets = torch.randint(0, NUM_MOVES, (32,))
    wdl_targets = torch.randint(0, 3, (32,))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    def loss_fn():
        policy, wdl = model(planes)
        return F.cross_entropy(policy, move_targets) + F.cross_entropy(wdl, wdl_targets)

    initial = loss_fn().item()
    for _ in range(150):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * 0.2, (initial, loss.item())


def test_onnx_export_matches_torch_with_dynamic_batch():
    model = ChessNet(ModelConfig.tiny()).eval()
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        (start_planes(1),),
        buffer,
        input_names=["planes"],
        output_names=["policy", "wdl"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}, "wdl": {0: "batch"}},
        opset_version=17,
    )
    session = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"])

    x = torch.randn(5, 18, 8, 8)  # a different batch size than the export example
    onnx_policy, onnx_wdl = session.run(None, {"planes": x.numpy()})
    with torch.no_grad():
        policy, wdl = model(x)
    np.testing.assert_allclose(onnx_policy, policy.numpy(), atol=1e-4)
    np.testing.assert_allclose(onnx_wdl, wdl.numpy(), atol=1e-4)
