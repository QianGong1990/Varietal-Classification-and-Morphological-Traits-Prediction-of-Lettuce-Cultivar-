import platform
import sys
from pathlib import Path

import torch

from training.data import dataset_summary_df
from training.gpu_utils import auto_batch_size, gpu_info_str

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def write_system_info(epochs: int = 50, patience: int = 10, batch_size: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bs = batch_size if batch_size is not None else auto_batch_size()
    lines = [
        f"Python version: {sys.version.split()[0]} ({platform.platform()})",
        f"PyTorch version: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}",
        f"GPU: {gpu_info_str()}",
        "Random seed: 42",
        "Train: 272 images",
        "Valid: 58 images",
        "Test: 58 images",
        "Augmented train RGB/segmented: 4352 images",
        "Augmented train masks: 1632 images",
        f"Batch size (auto-detected): {bs}",
        f"Max epochs: {epochs}",
        f"Early stopping patience: {patience}",
    ]
    with open(RESULTS_DIR / "system_info.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_dataset_summary():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = dataset_summary_df()
    df.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    write_system_info()
    write_dataset_summary()
