"""Computes real per-image predictions on the 58 held-out test images for:
- Regression: InceptionResNetV2 (rgb) -- best overall model
- Classification: SimpleCNN (segmented) -- ties for best, most efficient

Saves results/predictions/test_predictions.json for the dashboard's
interactive "Predict" explorer (real precomputed predictions, not live
in-browser inference).
"""
import json
from pathlib import Path

import torch

from training.data import (
    ARCH_RESOLUTION, REGRESSION_TARGETS, VARIETIES, build_manifest,
    LettuceDataset,
)
from training.gpu_utils import get_device
from training.models import build_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "predictions"


def main():
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "normalization_params.json") as f:
        norm_params = json.load(f)
    target_norm = norm_params["targets"]

    # ---- Regression: InceptionResNetV2 + rgb ----
    reg_arch, reg_input = "InceptionResNetV2", "rgb"
    reg_res = ARCH_RESOLUTION[reg_arch]
    reg_manifest = build_manifest(reg_input, reg_res)
    reg_test = reg_manifest[reg_manifest["split"] == "test"].reset_index(drop=True)

    reg_model = build_model(reg_arch, len(REGRESSION_TARGETS)).to(device)
    reg_model.load_state_dict(torch.load(
        RESULTS_DIR / "regression" / "model_weights" / f"{reg_arch}_{reg_input}_best.pth",
        map_location=device, weights_only=True))
    reg_model.eval()

    reg_ds = LettuceDataset(reg_test, reg_res, reg_input, "regression", pixel_norm=None, target_norm=target_norm)

    reg_preds = {}
    with torch.no_grad():
        for i in range(len(reg_ds)):
            x, y = reg_ds[i]
            out = reg_model(x.unsqueeze(0).to(device))[0].cpu().numpy()
            row = reg_test.iloc[i]
            preds = {}
            for j, trait in enumerate(REGRESSION_TARGETS):
                mean = target_norm[trait]["mean"]
                std = target_norm[trait]["std"]
                preds[trait] = float(out[j] * std + mean)
            reg_preds[row["filename"]] = preds

    # ---- Classification: SimpleCNN + segmented ----
    clf_arch, clf_input = "SimpleCNN", "segmented"
    clf_res = ARCH_RESOLUTION[clf_arch]
    clf_manifest = build_manifest(clf_input, clf_res)
    clf_test = clf_manifest[clf_manifest["split"] == "test"].reset_index(drop=True)

    pixel_norm = norm_params["pixel_norm"][f"{clf_input}_{clf_res}"]

    clf_model = build_model(clf_arch, len(VARIETIES)).to(device)
    clf_model.load_state_dict(torch.load(
        RESULTS_DIR / "classification" / "model_weights" / f"{clf_arch}_{clf_input}_best.pth",
        map_location=device, weights_only=True))
    clf_model.eval()

    clf_ds = LettuceDataset(clf_test, clf_res, clf_input, "classification", pixel_norm=pixel_norm)

    clf_preds = {}
    with torch.no_grad():
        for i in range(len(clf_ds)):
            x, y = clf_ds[i]
            out = clf_model(x.unsqueeze(0).to(device))[0]
            probs = torch.softmax(out, dim=0).cpu().numpy()
            row = clf_test.iloc[i]
            pred_idx = int(probs.argmax())
            clf_preds[row["filename"]] = {
                "predicted_variety": VARIETIES[pred_idx],
                "confidence": float(probs[pred_idx]),
            }

    # ---- Ground truth ----
    from training.data import load_split_df, load_ground_truth
    split_df = load_split_df()
    gt = load_ground_truth()
    test_rows = split_df[split_df["split"] == "test"]

    records = []
    for _, r in test_rows.iterrows():
        fname = r["filename"]
        image_id = int(r["id"])
        gt_row = gt[f"Image{image_id}"]
        rec = {
            "filename": fname,
            "variety": r["variety"],
            "true": {t: gt_row[t] for t in REGRESSION_TARGETS},
            "predicted": reg_preds.get(fname, {}),
            "classification": clf_preds.get(fname, {}),
        }
        records.append(rec)

    records.sort(key=lambda r: int(r["filename"].replace("RGB_", "").replace(".png", "")))

    out = {
        "regression_model": f"{reg_arch} ({reg_input})",
        "classification_model": f"{clf_arch} ({clf_input})",
        "n_test": len(records),
        "records": records,
    }
    with open(OUT_DIR / "test_predictions.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved {len(records)} test predictions to {OUT_DIR / 'test_predictions.json'}")
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main()
