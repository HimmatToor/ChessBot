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

## Layout

| Path | Purpose |
|---|---|
| `chessnet/` | Library code: board encoding, model, data loading |
| `tests/` | `pytest` tests |
| `data/`, `checkpoints/` | Downloaded games and trained weights (git-ignored) |
