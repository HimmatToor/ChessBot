# ChessBot training

Python code that trains the bot's neural network and exports it to ONNX for the web app.

## Local setup (macOS, CPU)

```bash
cd training
/opt/anaconda3/bin/python3.11 -m venv .venv   # any Python 3.11 works
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Local runs are CPU-only and meant for tests and small experiments.
Full training runs on a free Kaggle GPU notebook.

## Preparing data

Games come from the [Lichess Elite Database](https://database.nikonoel.fr/)
(Lichess games where both players are 2400+ vs 2200+; one month ≈ 85 MB zipped).

```bash
mkdir -p data/raw
curl -L -o data/raw/lichess_elite_2025-06.zip https://database.nikonoel.fr/lichess_elite_2025-06.zip
python prepare_data.py data/raw/lichess_elite_2025-06.zip --max-games 20000
```

This writes compressed shards to `data/processed/` (`train-*.npz`, `val-*.npz`).
On a 2020 Intel MacBook it processes ~135 games/s (~11k positions/s), and
stores ~11 bytes per position, so 20M positions ≈ 230 MB.
Run `python prepare_data.py --help` for the filters (rating, time control, skipped opening plies).

## Layout

| Path | Purpose |
|---|---|
| `chessnet/` | Library code: board encoding, model, data loading |
| `tests/` | `pytest` tests |
| `data/`, `checkpoints/` | Downloaded games and trained weights (git-ignored) |
