"""Combined 2x2 scatter panel of actual vs predicted for the best model per trait.

Panels (as reported): InceptionResNetV2+segmented for LeafArea, InceptionResNetV2+rgb
for Diameter and FreshWeightShoot, SimpleCNN+rgb for DryWeightShoot.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from training.data import (
    ARCH_RESOLUTION, LettuceDataset, REGRESSION_TARGETS, TRAIT_DISPLAY, TRAIT_UNIT,
    build_manifest,
)
from training.gpu_utils import get_device
from training.metrics import regression_metrics
from training.models import build_model
from training.plotting import CULTIVAR_COLORS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "figures"

# (trait, architecture, input_type) shown in the report.
PANELS = [
    ("LeafArea", "InceptionResNetV2", "segmented"),
    ("Diameter", "InceptionResNetV2", "rgb"),
    ("FreshWeightShoot", "InceptionResNetV2", "rgb"),
    ("DryWeightShoot", "SimpleCNN", "rgb"),
]

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, BOX_FS = 20, 18, 16, 15, 16


def predict(arch: str, input_type: str, norm_params: dict):
    """Returns (targets_denorm, preds_denorm, cultivars) on the test split."""
    device = get_device()
    resolution = ARCH_RESOLUTION[arch]
    target_norm = norm_params["targets"]
    pixel_norm = (norm_params["pixel_norm"][f"{input_type}_{resolution}"]
                  if arch == "SimpleCNN" else None)

    manifest = build_manifest(input_type, resolution)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    ds = LettuceDataset(test_df, resolution, input_type, "regression", pixel_norm, target_norm)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    model = build_model(arch, len(REGRESSION_TARGETS)).to(device)
    weight_path = RESULTS_DIR / "regression" / "model_weights" / f"{arch}_{input_type}_best.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mean = np.array([target_norm[t]["mean"] for t in REGRESSION_TARGETS])
    std = np.array([target_norm[t]["std"] for t in REGRESSION_TARGETS])
    return targets * std + mean, preds * std + mean, test_df["variety"].tolist()


def main():
    with open(RESULTS_DIR / "normalization_params.json") as f:
        norm_params = json.load(f)

    cache = {}
    for _, arch, input_type in PANELS:
        if (arch, input_type) not in cache:
            cache[(arch, input_type)] = predict(arch, input_type, norm_params)
            print(f"predicted: {arch} ({input_type})")

    fig, axes = plt.subplots(2, 2, figsize=(12, 11.5))
    for ax, (trait, arch, input_type) in zip(axes.ravel(), PANELS):
        targets, preds, cultivars = cache[(arch, input_type)]
        i = REGRESSION_TARGETS.index(trait)
        y_true, y_pred = targets[:, i], preds[:, i]
        cultivars = np.array(cultivars)

        for cultivar, color in CULTIVAR_COLORS.items():
            mask = cultivars == cultivar
            if mask.sum() == 0:
                continue
            ax.scatter(y_true[mask], y_pred[mask], color=color, label=cultivar,
                       alpha=0.8, edgecolor="k", linewidth=0.3, s=55)

        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.5, zorder=0)

        m = regression_metrics(y_true, y_pred)
        label = TRAIT_DISPLAY.get(trait, trait)
        unit = TRAIT_UNIT[trait]
        ax.set_title(f"{arch} - {input_type} - {label}", fontsize=TITLE_FS)
        ax.set_xlabel(f"Actual {label} ({unit})", fontsize=LABEL_FS)
        ax.set_ylabel(f"Predicted {label} ({unit})", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(fontsize=LEGEND_FS, loc="lower right", framealpha=0.9)
        ax.text(0.03, 0.97, f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.3f}",
                transform=ax.transAxes, fontsize=BOX_FS, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
        print(f"{arch} {input_type} {trait}: R2={m['R2']:.3f} RMSE={m['RMSE']:.3f}")

    fig.subplots_adjust(wspace=0.24, hspace=0.28, left=0.07, right=0.99, top=0.96, bottom=0.06)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        out = OUT_DIR / f"scatter_panel_best_models.{suffix}"
        fig.savefig(out, format=suffix, dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
