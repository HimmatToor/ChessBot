"""Kaggle training job for ChessBot (pushed with `kaggle kernels push -p training/kaggle`).

Clones the repo, trains on the attached dataset, exports chessnet.onnx.
Everything left in /kaggle/working becomes the run's downloadable output.

Edit RUN below for each run; the git history then records every config.
For a local dry run (CPU, tiny model) see training/README.md.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

RUN = {
    "name": "smoke",  # output folder name
    "model": "full",
    "batch_size": 1024,
    "steps": 1500,  # None = use "epochs"
    "epochs": 1,
    "lr": 1e-3,
    "warmup": 300,
    "eval_every": 500,
    "max_val": 50_000,
    "max_hours": 11.0,  # Kaggle stops sessions at 12 h; leave time to export
}

REPO = "https://github.com/HimmatToor/ChessBot.git"
INPUT = os.environ.get("CHESSBOT_INPUT", "/kaggle/input")
WORK = os.environ.get("CHESSBOT_WORK", "/kaggle/working")
# Set to a local checkout to dry-run without cloning.
LOCAL_REPO = os.environ.get("CHESSBOT_REPO")
REQUIRE_GPU = os.environ.get("CHESSBOT_REQUIRE_GPU", "1") == "1"


def sh(*cmd, cwd=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def find_data_dir():
    """Kaggle's mount path for datasets has changed over time, so search for the shards."""
    hits = sorted(glob.glob(os.path.join(INPUT, "**", "train-0000.npz"), recursive=True))
    if not hits:
        sys.exit(f"no train-0000.npz under {INPUT}: is the dataset attached?")
    return os.path.dirname(hits[0])


def main():
    start = time.time()
    import torch

    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        print(f"GPU {name} (compute capability {major}.{minor})", flush=True)
        # Fail fast if this torch build can't run on the GPU we were given (e.g. P100 on new torch).
        torch.zeros(1, device="cuda").add_(1).item()
    elif REQUIRE_GPU:
        sys.exit("no GPU: enable one in the notebook settings (needs a phone-verified account)")

    if LOCAL_REPO:
        repo = LOCAL_REPO
    else:
        repo = "/tmp/ChessBot"  # outside /kaggle/working so it isn't saved as output
        shutil.rmtree(repo, ignore_errors=True)
        sh("git", "clone", "--depth", "1", REPO, repo)
        sh("git", "log", "-1", "--format=commit %h %s", cwd=repo)
        # torch and numpy come with the Kaggle image; add the rest.
        sh(sys.executable, "-m", "pip", "install", "-q", "chess==1.11.2", "onnx==1.17.0", "onnxruntime==1.19.2")
    training = os.path.join(repo, "training")

    data = find_data_dir()
    out = os.path.join(WORK, RUN["name"])
    print(f"data {data}  ->  {out}", flush=True)

    args = [
        sys.executable, "train.py",
        "--data", data, "--out", out, "--model", RUN["model"],
        "--batch-size", str(RUN["batch_size"]), "--lr", str(RUN["lr"]), "--warmup", str(RUN["warmup"]),
        "--eval-every", str(RUN["eval_every"]), "--max-val", str(RUN["max_val"]),
        "--max-hours", str(RUN["max_hours"]),
    ]
    args += ["--steps", str(RUN["steps"])] if RUN["steps"] else ["--epochs", str(RUN["epochs"])]
    resume = os.path.join(out, "latest.pt")
    if os.path.exists(resume):
        args += ["--resume", resume]
    sh(*args, cwd=training)

    sh(sys.executable, "export_onnx.py", os.path.join(out, "best.pt"), "--out", os.path.join(WORK, "chessnet.onnx"),
       cwd=training)
    print(f"total time {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
