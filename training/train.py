import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from training.data import (
    ARCH_RESOLUTION, REGRESSION_TARGETS, TRAIT_UNIT, VARIETIES,
    LettuceDataset, build_manifest, compute_pixel_norm_stats, compute_target_norm_stats,
)
from training.gpu_utils import auto_batch_size, get_device
from training.metrics import regression_metrics
from training.models import build_model
from training.plotting import confusion_matrix_plot, scatter_plot, training_curve_plot
from training.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
NORM_PARAMS_PATH = RESULTS_DIR / "normalization_params.json"


def _update_norm_params(update: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    params = {}
    if NORM_PARAMS_PATH.exists():
        with open(NORM_PARAMS_PATH) as f:
            params = json.load(f)
    for k, v in update.items():
        params.setdefault(k, {})
        params[k].update(v)
    with open(NORM_PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)


def run(task: str, input_type: str, arch: str, epochs: int = 50, batch_size: int | None = None,
        seed: int = 42, patience: int = 10, lr: float = 1e-3, lr_factor: float = 0.3,
        lr_patience: int = 5, quick: bool = False, quick_n: int = 16) -> dict:
    assert task in ("classification", "regression")
    set_seed(seed)
    device = get_device()
    resolution = ARCH_RESOLUTION[arch]
    if batch_size is None:
        batch_size = auto_batch_size()

    manifest = build_manifest(input_type, resolution)
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    valid_df = manifest[manifest["split"] == "valid"].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    if quick:
        train_df = train_df.sample(n=min(quick_n, len(train_df)), random_state=seed).reset_index(drop=True)
        valid_df = valid_df.sample(n=min(quick_n, len(valid_df)), random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(n=min(quick_n, len(test_df)), random_state=seed).reset_index(drop=True)
        epochs = min(epochs, 2)

    pixel_norm = None
    target_norm = None
    if arch == "SimpleCNN":
        pixel_norm = compute_pixel_norm_stats(train_df, input_type, resolution)
        _update_norm_params({"pixel_norm": {f"{input_type}_{resolution}": pixel_norm}})

    if task == "regression":
        target_norm = compute_target_norm_stats(train_df)
        _update_norm_params({"targets": target_norm})

    train_ds = LettuceDataset(train_df, resolution, input_type, task, pixel_norm, target_norm)
    valid_ds = LettuceDataset(valid_df, resolution, input_type, task, pixel_norm, target_norm)
    test_ds = LettuceDataset(test_df, resolution, input_type, task, pixel_norm, target_norm)

    num_workers = 0 if quick else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    out_dim = len(VARIETIES) if task == "classification" else len(REGRESSION_TARGETS)
    model = build_model(arch, out_dim).to(device)

    loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=lr_factor, patience=lr_patience)

    run_name = f"{arch}_{input_type}"
    log_dir = RESULTS_DIR / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task}_{run_name}_log.csv"

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    early_stop_epoch = None

    history_rows = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, n_train = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
            n_train += x.size(0)
        train_loss = train_loss_sum / max(n_train, 1)

        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss_sum += loss.item() * x.size(0)
                n_val += x.size(0)
        val_loss = val_loss_sum / max(n_val, 1)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})
        print(f"[{run_name}] epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={current_lr:.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop_epoch = epoch
                print(f"[{run_name}] early stopping at epoch {epoch} (best={best_epoch})")
                break

    train_seconds = time.time() - t0

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(history_rows)

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history_rows)
    fig_dir = RESULTS_DIR / "figures" / "training_curves"
    training_curve_plot(history_df, f"{task}_{arch}", input_type, best_epoch, early_stop_epoch, fig_dir)

    # ---- weight saving ----
    if task == "regression":
        weight_dir = RESULTS_DIR / "regression" / "model_weights"
    else:
        weight_dir = RESULTS_DIR / "classification" / "model_weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weight_dir / f"{run_name}_best.pth"
    torch.save(model.state_dict(), weight_path)

    # ---- test-set evaluation ----
    model.eval()
    all_preds, all_targets, all_filenames = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x).cpu().numpy()
            all_preds.append(out)
            all_targets.append(y.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    filenames = test_df["filename"].tolist()
    cultivars = test_df["variety"].tolist()

    result = {"task": task, "input_type": input_type, "architecture": arch, "resolution": resolution,
              "n_train": len(train_df), "n_valid": len(valid_df), "n_test": len(test_df),
              "batch_size": batch_size, "best_epoch": best_epoch, "epochs_ran": len(history_rows),
              "train_seconds": train_seconds}

    if task == "classification":
        pred_labels = all_preds.argmax(axis=1)
        acc = float((pred_labels == all_targets).mean())
        cm = confusion_matrix(all_targets, pred_labels, labels=list(range(len(VARIETIES))))
        report = classification_report(all_targets, pred_labels, target_names=VARIETIES, digits=4)

        clf_dir = RESULTS_DIR / "classification"
        clf_dir.mkdir(parents=True, exist_ok=True)
        with open(clf_dir / f"classification_report_{arch}_{input_type}.txt", "w") as f:
            f.write(report)

        cls_results_path = clf_dir / "classification_results.csv"
        row = {"model": arch, "input_type": input_type, "accuracy": acc, "n_test": len(test_df)}
        _upsert_csv_row(cls_results_path, row, key_cols=["model", "input_type"])

        confusion_matrix_plot(cm, VARIETIES, RESULTS_DIR / "figures" / "confusion_matrix",
                              model=arch, input_type=input_type)
        shutil.copy(RESULTS_DIR / "figures" / "confusion_matrix" / f"confusion_matrix_{arch}_{input_type}.png",
                    clf_dir / f"confusion_matrix_{arch}_{input_type}.png")

        result["accuracy"] = acc
    else:
        # denormalize
        mean = np.array([target_norm[t]["mean"] for t in REGRESSION_TARGETS])
        std = np.array([target_norm[t]["std"] for t in REGRESSION_TARGETS])
        preds_denorm = all_preds * std + mean
        targets_denorm = all_targets * std + mean

        reg_dir = RESULTS_DIR / "regression"
        scatter_dir = reg_dir / "scatter_plots"
        fig_scatter_dir = RESULTS_DIR / "figures" / "scatter_plots"
        scatter_dir.mkdir(parents=True, exist_ok=True)

        all_results_path = reg_dir / "all_results.csv"
        per_cultivar_path = reg_dir / "per_cultivar_results.csv"

        per_trait_metrics = {}
        for i, trait in enumerate(REGRESSION_TARGETS):
            y_true = targets_denorm[:, i]
            y_pred = preds_denorm[:, i]
            m = regression_metrics(y_true, y_pred)
            per_trait_metrics[trait] = m
            _append_csv_row(all_results_path, {"model": arch, "input_type": input_type, "trait": trait, **m})

            scatter_plot(y_true, y_pred, cultivars, trait, TRAIT_UNIT[trait], arch, input_type,
                        m["R2"], m["RMSE"], fig_scatter_dir)
            shutil.copy(fig_scatter_dir / f"{arch}_{input_type}_{trait}.png",
                        scatter_dir / f"{arch}_{input_type}_{trait}.png")

            for cultivar in VARIETIES:
                mask = np.array(cultivars) == cultivar
                if mask.sum() < 2:
                    continue
                cm_ = regression_metrics(y_true[mask], y_pred[mask])
                _append_csv_row(per_cultivar_path, {
                    "model": arch, "input_type": input_type, "trait": trait, "cultivar": cultivar,
                    "R2": cm_["R2"], "RMSE": cm_["RMSE"],
                })

        result["per_trait_metrics"] = per_trait_metrics

    print(f"[{run_name}] DONE in {train_seconds:.1f}s")
    return result


def _append_csv_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def _upsert_csv_row(path: Path, row: dict, key_cols: list):
    """Like _append_csv_row, but replaces any existing row matching key_cols
    instead of appending a duplicate (safe to re-run a single combo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists():
        df = pd.read_csv(path)
        mask = pd.Series(True, index=df.index)
        for col in key_cols:
            mask &= (df[col] == row[col])
        df = df[~mask]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--input_type", required=True, choices=["rgb", "segmented", "mask"])
    ap.add_argument("--arch", required=True, choices=list(ARCH_RESOLUTION.keys()))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--quick_n", type=int, default=16)
    args = ap.parse_args()

    run(args.task, args.input_type, args.arch, args.epochs, args.batch_size,
        args.seed, args.patience, args.lr, quick=args.quick, quick_n=args.quick_n)


if __name__ == "__main__":
    main()
