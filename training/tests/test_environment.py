"""Smoke tests: the toolchain the training pipeline relies on works end to end."""

import io

import chess
import numpy as np
import onnxruntime as ort
import torch


def test_python_chess_generates_legal_moves():
    assert len(list(chess.Board().legal_moves)) == 20


def test_torch_forward_and_backward():
    model = torch.nn.Sequential(torch.nn.Conv2d(18, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Flatten())
    out = model(torch.zeros(2, 18, 8, 8))
    out.sum().backward()
    assert out.shape == (2, 8 * 8 * 8)


def test_torch_model_runs_in_onnxruntime():
    model = torch.nn.Sequential(torch.nn.Conv2d(18, 4, 1), torch.nn.Flatten()).eval()
    x = torch.randn(1, 18, 8, 8)
    buffer = io.BytesIO()
    torch.onnx.export(model, (x,), buffer, input_names=["planes"], output_names=["out"], opset_version=17)

    session = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"])
    (onnx_out,) = session.run(None, {"planes": x.numpy()})
    np.testing.assert_allclose(onnx_out, model(x).detach().numpy(), atol=1e-5)
