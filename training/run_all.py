"""Driver: 15 regression runs (3 input types x 5 architectures) +
1 classification run (SimpleCNN, segmented only) = 16 total runs."""
import argparse
import json
from pathlib import Path

from training.data import ARCH_RESOLUTION
from training.system_info import write_dataset_summary, write_system_info
from training.train import RESULTS_DIR, run

ARCHITECTURES = list(ARCH_RESOLUTION.keys())
REGRESSION_INPUTS = ["rgb", "segmented", "mask"]


def build_plan():
    plan = []
    for input_type in REGRESSION_INPUTS:
        for arch in ARCHITECTURES:
            plan.append({"task": "regression", "input_type": input_type, "arch": arch})
    plan.append({"task": "classification", "input_type": "segmented", "arch": "SimpleCNN"})
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    write_system_info(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
    write_dataset_summary()

    plan = build_plan()
    print(f"Total runs planned: {len(plan)} (expected 16)")
    assert len(plan) == 16

    results = []
    for i, cfg in enumerate(plan, 1):
        print(f"\n=== [{i}/{len(plan)}] {cfg['task']} / {cfg['input_type']} / {cfg['arch']} ===")
        try:
            summary = run(cfg["task"], cfg["input_type"], cfg["arch"], epochs=args.epochs,
                          batch_size=args.batch_size, seed=args.seed, patience=args.patience,
                          lr=args.lr, quick=args.quick)
            summary["status"] = "ok"
        except Exception as e:
            summary = {**cfg, "status": "failed", "error": str(e)}
            print(f"FAILED: {e}")
        results.append(summary)

    out_path = RESULTS_DIR / "all_runs_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll-run summary saved to {out_path}")

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"Completed: {ok}/{len(plan)}")


if __name__ == "__main__":
    main()
