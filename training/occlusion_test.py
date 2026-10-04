"""Causal occlusion test: does zeroing the PLANT vs zeroing the BACKGROUND
actually change the model's prediction? This is the causal follow-up to the
Grad-CAM analysis (which only shows gradient magnitude, not causal reliance).

Uses the binary mask (pixel-aligned to the RGB/segmented crop+resize pipeline)
to precisely zero out plant-only or background-only regions.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from training.data import ARCH_RESOLUTION, DATASET_ROOT, REGRESSION_TARGETS, TRAIT_DISPLAY, TRAIT_UNIT, VARIETIES
from training.gpu_utils import get_device
from training.models import build_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "occlusion"

REP_IMAGES = {
    "Aphylion": "RGB_46.png",
    "Lugano": "RGB_68.png",
    "Salanova": "RGB_92.png",
    "Satine": "RGB_4.png",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_raw_and_mask(rgb_path: str, mask_path: str, resolution: int):
    img = Image.open(rgb_path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]

    mask_img = Image.open(mask_path).convert("L")
    if mask_img.size != (resolution, resolution):
        mask_img = mask_img.resize((resolution, resolution), Image.NEAREST)
    mask = np.asarray(mask_img) > 127  # True = plant

    return arr, mask, img


def normalize(arr: np.ndarray, pixel_norm: dict | None) -> torch.Tensor:
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if pixel_norm is not None:
        mean = torch.tensor(pixel_norm["mean"], dtype=torch.float32)
        std = torch.tensor(pixel_norm["std"], dtype=torch.float32)
    else:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32)
    return (tensor - mean[:, None, None]) / std[:, None, None]


def occlude(arr: np.ndarray, mask: np.ndarray, keep: str) -> np.ndarray:
    """keep='plant' zeros background, keeping only plant pixels.
    keep='background' zeros the plant, keeping only background pixels."""
    out = arr.copy()
    if keep == "plant":
        out[~mask] = 0.0
    elif keep == "background":
        out[mask] = 0.0
    return out


def run_regression_occlusion(arch: str, input_type: str, trait: str, weight_path: Path,
                              pixel_norm: dict | None = None):
    device = get_device()
    resolution = ARCH_RESOLUTION[arch]
    base_name = {"rgb": "RGBImages", "segmented": "removed_background", "mask": "binary_masks"}[input_type]
    img_dir = DATASET_ROOT / f"{base_name}_{resolution}"
    mask_dir = DATASET_ROOT / f"binary_masks_{resolution}"

    with open(RESULTS_DIR / "normalization_params.json") as f:
        norm_params = json.load(f)
    target_mean = norm_params["targets"][trait]["mean"]
    target_std = norm_params["targets"][trait]["std"]
    trait_idx = REGRESSION_TARGETS.index(trait)

    model = build_model(arch, len(REGRESSION_TARGETS)).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    results = []
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for col, (cultivar, fname) in enumerate(REP_IMAGES.items()):
        arr, mask, pil_img = load_raw_and_mask(str(img_dir / fname), str(mask_dir / fname), resolution)

        variants = {
            "original": arr,
            "plant only\n(bg zeroed)": occlude(arr, mask, keep="plant"),
            "background only\n(plant zeroed)": occlude(arr, mask, keep="background"),
        }

        preds = {}
        for name, variant_arr in variants.items():
            x = normalize(variant_arr, pixel_norm).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)[0, trait_idx].item()
            pred_denorm = out * target_std + target_mean
            preds[name] = pred_denorm

        results.append({"cultivar": cultivar, **{k: v for k, v in preds.items()}})

        for row, (name, variant_arr) in enumerate(variants.items()):
            axes[row, col].imshow(variant_arr)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(cultivar, fontsize=12)
            axes[row, col].text(0.5, -0.08, f"{name}: {preds[name]:.2f} {TRAIT_UNIT[trait]}",
                                 transform=axes[row, col].transAxes, ha="center", fontsize=10)

    fig.suptitle(f"Causal Occlusion Test: {arch} ({input_type}) — {TRAIT_DISPLAY.get(trait, trait)}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"occlusion_{arch}_{input_type}_{trait}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    print(f"\n=== {arch} / {input_type} / {trait} ===")
    for r in results:
        orig = r["original"]
        plant_only = r["plant only\n(bg zeroed)"]
        bg_only = r["background only\n(plant zeroed)"]
        print(f"{r['cultivar']:10s}  original={orig:8.2f}  plant_only={plant_only:8.2f} (Δ={plant_only-orig:+7.2f})  "
              f"bg_only={bg_only:8.2f} (Δ={bg_only-orig:+7.2f})")

    return results


if __name__ == "__main__":
    weight_dir = RESULTS_DIR / "regression" / "model_weights"

    print("############ InceptionResNetV2 + RGB ############")
    for trait in REGRESSION_TARGETS:
        run_regression_occlusion("InceptionResNetV2", "rgb", trait,
                                  weight_dir / "InceptionResNetV2_rgb_best.pth")

    print("\n############ SimpleCNN + RGB (DryWeightShoot) ############")
    with open(RESULTS_DIR / "normalization_params.json") as f:
        norm_params = json.load(f)
    pixel_norm = norm_params["pixel_norm"]["rgb_224"]
    run_regression_occlusion("SimpleCNN", "rgb", "DryWeightShoot",
                              weight_dir / "SimpleCNN_rgb_best.pth", pixel_norm=pixel_norm)
