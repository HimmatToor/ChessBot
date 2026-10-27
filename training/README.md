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
It uses every CPU core (`--workers`, output is identical for any value): on a
4-core 2020 Intel MacBook ~420 games/s, so one month (~289k games, ~25M
positions) takes ~12 minutes. Shards take ~11 bytes per position (~275 MB per month).
Run `python prepare_data.py --help` for the filters (rating, time control, skipped opening plies).

## Training

```bash
python train.py --model tiny --batch-size 256 --epochs 1 --out checkpoints/tiny-cpu   # ~6 min on CPU
```

Prints validation move-prediction accuracy and win/draw/loss accuracy every
`--eval-every` steps and saves `latest.pt`, `best.pt` and `metrics.csv` to `--out`.
Resume an interrupted run with `--resume checkpoints/<run>/latest.pt`.
`--model full` is the 10×128 network meant for a GPU.

## Training on Kaggle (free GPU)

`kaggle/run.py` is a Kaggle script job: it clones this repo from GitHub (so
**push your commits first**) and trains on the attached dataset. Export the
downloaded `best.pt` locally with `export_onnx.py` (Kaggle's newer torch uses a
different ONNX exporter; the local pinned toolchain is the tested one). Its settings live in the `RUN` dict at the top; the dataset and
GPU type in `kaggle/kernel-metadata.json`.

```bash
kaggle kernels push -p kaggle                    # starts a run (new version)
kaggle kernels status himmattoor/chessbot-train  # queued / running / complete / error
kaggle kernels output himmattoor/chessbot-train -p runs/latest   # log + checkpoints + onnx
```

Dry-run the script locally (CPU, after shrinking `RUN` to the tiny model):

```bash
CHESSBOT_INPUT=data CHESSBOT_WORK=/tmp/kaggle-work CHESSBOT_REPO=$PWD/.. \
  CHESSBOT_REQUIRE_GPU=0 python kaggle/run.py
```

## Measuring strength

`evaluate.py` plays each difficulty level against Stockfish at fixed ratings and
estimates the level's rating. It needs a Stockfish binary. Homebrew has no
prebuilt package for Intel Macs on recent macOS (it compiles from source, which
fails without working Command Line Tools), so use the official build:

```bash
mkdir -p tools/stockfish-19 && cd tools/stockfish-19
curl -LO https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-macos-universal.tar.gz
tar -xzf stockfish-macos-universal.tar.gz && cd ../..
python evaluate.py runs/full-v1/full-v1/best.pt --levels medium hard --opponents 1400 1700 2000 --games 20
```

`evaluate.py` finds it there automatically (or uses `stockfish` on PATH).

## Exporting for the website

```bash
python export_onnx.py checkpoints/tiny-cpu/best.pt   # -> ../web/public/models/chessnet.onnx
```

Stores weights as float16 (tiny model 2 MB, full model 14 MB, under Cloudflare
Pages' 25 MB file limit) and checks onnxruntime's output against PyTorch.

## Layout

| Path | Purpose |
|---|---|
| `chessnet/` | Library code: board encoding, model, data loading |
| `tests/` | `pytest` tests |
| `data/`, `checkpoints/` | Downloaded games and trained weights (git-ignored) |
