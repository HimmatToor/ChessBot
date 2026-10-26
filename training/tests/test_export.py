import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

import export_onnx
from chessnet.checkpoint import save_checkpoint
from chessnet.model import ChessNet, ModelConfig


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    model = ChessNet(ModelConfig.tiny())
    # Non-trivial BatchNorm statistics, so folding them into the convs is actually exercised.
    model.train()
    with torch.no_grad():
        for _ in range(3):
            model(torch.rand(64, 18, 8, 8))
    path = tmp_path / "best.pt"
    save_checkpoint(path, model.eval(), step=123)
    return path


@pytest.mark.parametrize("fp16", [True, False])
def test_export_matches_pytorch(checkpoint, tmp_path, fp16):
    out = tmp_path / "model.onnx"
    stats = export_onnx.export(checkpoint, out, fp16=fp16)
    assert out.exists()
    assert stats["top1_agreement"] == 1.0
    tolerance = 5e-2 if fp16 else 1e-4
    assert stats["max_policy_diff"] < tolerance and stats["max_wdl_diff"] < tolerance


def test_fp16_halves_the_file(checkpoint, tmp_path):
    export_onnx.export(checkpoint, tmp_path / "fp16.onnx", fp16=True)
    export_onnx.export(checkpoint, tmp_path / "fp32.onnx", fp16=False)
    ratio = (tmp_path / "fp16.onnx").stat().st_size / (tmp_path / "fp32.onnx").stat().st_size
    assert 0.45 < ratio < 0.6


def test_exported_interface_and_metadata(checkpoint, tmp_path):
    out = tmp_path / "model.onnx"
    export_onnx.export(checkpoint, out)
    model = onnx.load(out)
    meta = {p.key: p.value for p in model.metadata_props}
    assert meta["encoding_version"] == "1"
    assert meta["num_moves"] == "1858"
    assert meta["training_step"] == "123"
    assert json.loads(meta["model_config"])["blocks"] == 2

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["planes"]
    assert [o.name for o in session.get_outputs()] == ["policy", "wdl"]
    policy, wdl = session.run(None, {"planes": np.zeros((7, 18, 8, 8), dtype=np.float32)})
    assert policy.shape == (7, 1858) and wdl.shape == (7, 3)
    assert policy.dtype == np.float32
