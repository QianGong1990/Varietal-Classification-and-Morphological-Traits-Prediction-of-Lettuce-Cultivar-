"""5-fold cross-validation (stratified by cultivar) on ALL 388 original
(non-augmented) removed_background images. Each fold's training images are
augmented on-the-fly (in-memory, 16 manuscript transforms incl. original)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from training.data import (
    ARCH_RESOLUTION, DATASET_ROOT, IMAGENET_MEAN, IMAGENET_STD, REGRESSION_TARGETS,
    load_ground_truth, load_split_df,
)
from training.gpu_utils import auto_batch_size, get_device
from training.metrics import regression_metrics
from training.models import build_model
from training.plotting import kfold_comparison_plot
from training.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SEED = 42
N_FOLDS = 5


def adjust_hue(img: Image.Image, delta: float) -> Image.Image:
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    h_arr = np.array(h).astype(np.int16)
    shift = int(round(delta * 255))
    h_arr = (h_arr + shift) % 256
    h_new = Image.fromarray(h_arr.astype(np.uint8), mode="L")
    return Image.merge("HSV", (h_new, s, v)).convert("RGB")


TRANSFORMS = {
    "original": lambda im: im,
    "brightness_08": lambda im: ImageEnhance.Brightness(im).enhance(0.8),
    "brightness_09": lambda im: ImageEnhance.Brightness(im).enhance(0.9),
    "brightness_11": lambda im: ImageEnhance.Brightness(im).enhance(1.1),
    "brightness_12": lambda im: ImageEnhance.Brightness(im).enhance(1.2),
    "hue_neg": lambda im: adjust_hue(im, -0.3),
    "hue_pos": lambda im: adjust_hue(im, 0.3),
    "sat_low": lambda im: ImageEnhance.Color(im).enhance(0.5),
    "sat_high": lambda im: ImageEnhance.Color(im).enhance(1.5),
    "contrast_15": lambda im: ImageEnhance.Contrast(im).enhance(1.5),
    "contrast_26": lambda im: ImageEnhance.Contrast(im).enhance(2.6),
    "rot90": lambda im: im.rotate(90, expand=False),
    "rot180": lambda im: im.rotate(180),
    "rot270": lambda im: im.rotate(270, expand=False),
    "hflip": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "vflip": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
}
TRANSFORM_NAMES = list(TRANSFORMS.keys())


def to_tensor_normalized(img: Image.Image, resolution: int) -> torch.Tensor:
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.BILINEAR)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


class FoldTrainDataset(Dataset):
    """On-the-fly augmentation: 16 transform variants per base image."""

    def __init__(self, filepaths, targets, resolution):
        self.filepaths = filepaths
        self.targets = targets
        self.resolution = resolution
        self.n_transforms = len(TRANSFORM_NAMES)

    def __len__(self):
        return len(self.filepaths) * self.n_transforms

    def __getitem__(self, idx):
        img_idx, t_idx = divmod(idx, self.n_transforms)
        img = Image.open(self.filepaths[img_idx]).convert("RGB")
        transform = TRANSFORMS[TRANSFORM_NAMES[t_idx]]
        img = transform(img)
        tensor = to_tensor_normalized(img, self.resolution)
        target = torch.tensor(self.targets[img_idx], dtype=torch.float32)
        return tensor, target


class PlainDataset(Dataset):
    def __init__(self, filepaths, targets, resolution):
        self.filepaths = filepaths
        self.targets = targets
        self.resolution = resolution

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        tensor = to_tensor_normalized(img, self.resolution)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return tensor, target


def train_one_fold(arch: str, fold: int, train_paths, train_targets, test_paths, test_targets,
                    target_mean, target_std, epochs: int, patience: int, lr: float,
                    batch_size: int | None, seed: int) -> dict:
    set_seed(seed + fold)
    device = get_device()
    resolution = ARCH_RESOLUTION[arch]
    if batch_size is None:
        batch_size = auto_batch_size()

    train_targets_norm = (np.array(train_targets) - target_mean) / target_std
    test_targets_norm = (np.array(test_targets) - target_mean) / target_std

    train_ds = FoldTrainDataset(train_paths, train_targets_norm, resolution)
    test_ds = PlainDataset(test_paths, test_targets_norm, resolution)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model(arch, out_dim=len(REGRESSION_TARGETS)).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum, n = 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss_sum += loss_fn(out, y).item() * x.size(0)
                n += x.size(0)
        val_loss = val_loss_sum / max(n, 1)
        scheduler.step(val_loss)
        print(f"[{arch} fold{fold}] epoch {epoch}/{epochs} val_loss={val_loss:.4f}")

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{arch} fold{fold}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            preds.append(model(x.to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0) * target_std + target_mean
    y_true = np.array(test_targets)

    metrics_per_trait = {}
    for i, trait in enumerate(REGRESSION_TARGETS):
        metrics_per_trait[trait] = regression_metrics(y_true[:, i], preds[:, i])
    return metrics_per_trait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    split_df = load_split_df()
    gt = load_ground_truth()

    filenames = split_df["filename"].values
    varieties = split_df["variety"].values
    ids = split_df["id"].values

    targets_all = np.array([[gt[f"Image{i}"][t] for t in REGRESSION_TARGETS] for i in ids])

    seg_dirs = {arch: DATASET_ROOT / f"removed_background_{ARCH_RESOLUTION[arch]}" for arch in ARCH_RESOLUTION}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    out_dir = RESULTS_DIR / "kfold"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for arch in ARCH_RESOLUTION:
        seg_dir = seg_dirs[arch]
        paths_all = np.array([str(seg_dir / f) for f in filenames])

        for fold, (train_idx, test_idx) in enumerate(skf.split(paths_all, varieties), 1):
            if args.quick and fold > 1:
                break
            train_paths = paths_all[train_idx]
            test_paths = paths_all[test_idx]
            train_targets = targets_all[train_idx]
            test_targets = targets_all[test_idx]

            if args.quick:
                train_paths = train_paths[:8]
                train_targets = train_targets[:8]
                test_paths = test_paths[:8]
                test_targets = test_targets[:8]

            target_mean = train_targets.mean(axis=0)
            target_std = train_targets.std(axis=0)
            target_std[target_std == 0] = 1.0

            epochs = min(args.epochs, 2) if args.quick else args.epochs

            print(f"\n=== {arch} fold {fold}/{N_FOLDS} ===")
            metrics_per_trait = train_one_fold(
                arch, fold, train_paths, train_targets, test_paths, test_targets,
                target_mean, target_std, epochs, args.patience, args.lr, args.batch_size, args.seed,
            )
            for trait, m in metrics_per_trait.items():
                fold_rows.append({"model": arch, "fold": fold, "trait": trait, **m})

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "kfold_results.csv", index=False)

    summary_rows = []
    for (model, trait), grp in fold_df.groupby(["model", "trait"]):
        summary_rows.append({
            "model": model, "trait": trait,
            "R2_mean": grp["R2"].mean(), "R2_std": grp["R2"].std(),
            "RMSE_mean": grp["RMSE"].mean(), "RMSE_std": grp["RMSE"].std(),
            "NRMSE_mean": grp["NRMSE"].mean(), "NRMSE_std": grp["NRMSE"].std(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "kfold_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    if len(summary_df):
        kfold_comparison_plot(summary_df, RESULTS_DIR / "figures" / "kfold")


if __name__ == "__main__":
    main()
