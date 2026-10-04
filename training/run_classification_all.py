"""Runs cultivar classification for all 5 architectures (segmented input)."""
from training.data import ARCH_RESOLUTION
from training.train import run

ARCHITECTURES = list(ARCH_RESOLUTION.keys())

if __name__ == "__main__":
    for arch in ARCHITECTURES:
        print(f"\n=== classification / segmented / {arch} ===")
        run("classification", "segmented", arch, epochs=50, seed=42, patience=10, lr=1e-3)
