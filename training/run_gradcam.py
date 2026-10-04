"""Runs Grad-CAM for the models most relevant to the reviewer's interpretability
question: does the winning RGB-input model actually attend to the plant, or to
the background/tray it was hypothesized to (harmlessly) retain?
"""
import json
from pathlib import Path

import torch

from training.data import (
    ARCH_RESOLUTION, DATASET_ROOT, REGRESSION_TARGETS, VARIETIES, load_split_df,
)
from training.gpu_utils import get_device
from training.gradcam import find_last_spatial_module_name, run_gradcam_grid
from training.models import build_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "gradcam"

# One representative test image per cultivar, used consistently across all figures.
REP_IMAGES = {
    "Aphylion": "RGB_46.png",
    "Lugano": "RGB_68.png",
    "Salanova": "RGB_92.png",
    "Satine": "RGB_4.png",
}


def image_paths_for(input_type: str, resolution: int):
    base_name = {"rgb": "RGBImages", "segmented": "removed_background", "mask": "binary_masks"}[input_type]
    d = DATASET_ROOT / f"{base_name}_{resolution}"
    paths = [str(d / fn) for fn in REP_IMAGES.values()]
    titles = list(REP_IMAGES.keys())
    return paths, titles


def get_predicted_classes(arch: str, input_type: str, image_paths: list, weight_path: Path):
    from training.gradcam import load_image_tensor
    device = get_device()
    resolution = ARCH_RESOLUTION[arch]
    model = build_model(arch, len(VARIETIES)).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    pixel_norm = None
    if arch == "SimpleCNN":
        with open(RESULTS_DIR / "normalization_params.json") as f:
            norm_params = json.load(f)
        pixel_norm = norm_params["pixel_norm"][f"{input_type}_{resolution}"]

    preds = []
    with torch.no_grad():
        for p in image_paths:
            x, _ = load_image_tensor(p, resolution, pixel_norm)
            out = model(x.to(device))
            preds.append(int(out.argmax(dim=1).item()))
    return preds


def main():
    # --- 1. Best overall regression model: InceptionResNetV2 + rgb, one figure per trait ---
    arch, input_type, task = "InceptionResNetV2", "rgb", "regression"
    resolution = ARCH_RESOLUTION[arch]
    image_paths, titles = image_paths_for(input_type, resolution)
    weight_path = RESULTS_DIR / "regression" / "model_weights" / f"{arch}_{input_type}_best.pth"

    for i, trait in enumerate(REGRESSION_TARGETS):
        out_path = OUT_DIR / f"gradcam_{arch}_{input_type}_{trait}.png"
        run_gradcam_grid(arch, input_type, task, image_paths, titles, i, trait, out_path, weight_path)

    # --- 2. SimpleCNN + rgb, the trait it uniquely wins (DryWeightShoot) ---
    arch, input_type, task = "SimpleCNN", "rgb", "regression"
    resolution = ARCH_RESOLUTION[arch]
    image_paths, titles = image_paths_for(input_type, resolution)
    weight_path = RESULTS_DIR / "regression" / "model_weights" / f"{arch}_{input_type}_best.pth"

    with open(RESULTS_DIR / "normalization_params.json") as f:
        norm_params = json.load(f)
    pixel_norm = norm_params["pixel_norm"][f"{input_type}_{resolution}"]

    trait_idx = REGRESSION_TARGETS.index("DryWeightShoot")
    out_path = OUT_DIR / f"gradcam_{arch}_{input_type}_DryWeightShoot.png"
    run_gradcam_grid(arch, input_type, task, image_paths, titles, trait_idx, "DryWeightShoot",
                      out_path, weight_path, pixel_norm=pixel_norm)

    # --- 3. Best/most efficient classification model: SimpleCNN + segmented ---
    arch, input_type, task = "SimpleCNN", "segmented", "classification"
    resolution = ARCH_RESOLUTION[arch]
    image_paths, titles = image_paths_for(input_type, resolution)
    weight_path = RESULTS_DIR / "classification" / "model_weights" / f"{arch}_{input_type}_best.pth"

    pixel_norm = norm_params["pixel_norm"][f"{input_type}_{resolution}"]
    pred_classes = get_predicted_classes(arch, input_type, image_paths, weight_path)
    titles_with_pred = [f"{c}\npred: {VARIETIES[p]}" for c, p in zip(titles, pred_classes)]

    out_path = OUT_DIR / f"gradcam_{arch}_{input_type}_classification.png"
    run_gradcam_grid(arch, input_type, task, image_paths, titles_with_pred, pred_classes,
                      "Predicted Cultivar", out_path, weight_path, pixel_norm=pixel_norm)


if __name__ == "__main__":
    main()
