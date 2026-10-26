import csv
import zipfile

import pytest
import torch

import prepare_data
import train
from chessnet.checkpoint import load_checkpoint


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    """A few hundred positions of real-looking games, written by prepare_data."""
    root = tmp_path_factory.mktemp("data")
    game = ('[Result "1-0"]\n[WhiteElo "2400"]\n[BlackElo "2400"]\n[TimeControl "300+0"]\n\n'
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 1-0\n\n")
    archive = root / "games.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("games.pgn", game * 40)
    prepare_data.main([str(archive), "--out", str(root / "processed"), "--skip-plies", "0", "--val-every", "10"])
    return root / "processed"


def read_metrics(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_lr_schedule_warms_up_then_decays():
    assert train.lr_at(0, 1000, 1e-3, 100) == pytest.approx(1e-5)
    assert train.lr_at(99, 1000, 1e-3, 100) == pytest.approx(1e-3)
    assert train.lr_at(999, 1000, 1e-3, 100) == pytest.approx(5e-5, rel=0.05)


def test_training_learns_and_writes_checkpoints(shards, tmp_path):
    out = tmp_path / "run"
    train.main(["--data", str(shards), "--out", str(out), "--model", "tiny", "--batch-size", "32",
                "--steps", "120", "--eval-every", "40", "--warmup", "10", "--lr", "3e-3", "--device", "cpu"])

    rows = read_metrics(out / "metrics.csv")
    assert [int(r["step"]) for r in rows] == [40, 80, 120]
    # The same 16-move game repeated: the model should learn to predict it.
    assert float(rows[-1]["val_policy_acc"]) > 0.9
    assert float(rows[-1]["val_policy_loss"]) < float(rows[0]["val_policy_loss"])

    model, checkpoint = load_checkpoint(out / "best.pt")
    assert checkpoint["step"] == 120
    policy, wdl = model(torch.zeros(1, 18, 8, 8))
    assert policy.shape == (1, 1858) and wdl.shape == (1, 3)


def test_resume_continues_from_saved_step(shards, tmp_path):
    out = tmp_path / "run"
    common = ["--data", str(shards), "--out", str(out), "--model", "tiny", "--batch-size", "32",
              "--eval-every", "20", "--device", "cpu"]
    train.main(common + ["--steps", "20"])
    train.main(common + ["--steps", "40", "--resume", str(out / "latest.pt")])

    assert [int(r["step"]) for r in read_metrics(out / "metrics.csv")] == [20, 40]
    _, checkpoint = load_checkpoint(out / "latest.pt")
    assert checkpoint["step"] == 40
