"""Dataset loading for the PyTorch lettuce training pipeline.

Ground truth: dataset/GroundTruth_All_388_Images.json
Split:        dataset/split/dataset_split.xlsx
Aug log:      dataset/augmented/augmentation_log.csv

Train split -> uses augmented files (all transform variants), matched to
               ground truth via augmentation_log.csv's original_filename.
Valid/test  -> uses the original (non-augmented) resized folders.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
AUG_ROOT = DATASET_ROOT / "augmented"

SPLIT_XLSX = DATASET_ROOT / "split" / "dataset_split.xlsx"
GT_JSON = DATASET_ROOT / "GroundTruth_All_388_Images.json"
AUG_LOG_CSV = AUG_ROOT / "augmentation_log.csv"

VARIETIES = ["Aphylion", "Lugano", "Salanova", "Satine"]
VARIETY_TO_IDX = {v: i for i, v in enumerate(VARIETIES)}

# Height removed per manuscript: only LA, D, FW, DW
REGRESSION_TARGETS = ["LeafArea", "Diameter", "FreshWeightShoot", "DryWeightShoot"]
TRAIT_SHORT = {"LeafArea": "LA", "Diameter": "D", "FreshWeightShoot": "FW", "DryWeightShoot": "DW"}
TRAIT_UNIT = {"LeafArea": "cm²", "Diameter": "cm", "FreshWeightShoot": "g", "DryWeightShoot": "g"}
TRAIT_DISPLAY = {"LeafArea": "Leaf Area", "Diameter": "Diameter", "FreshWeightShoot": "Fresh Weight",
                  "DryWeightShoot": "Dry Weight"}

INPUT_FOLDER_BASENAME = {
    "rgb": "RGBImages",
    "segmented": "removed_background",
    "mask": "binary_masks",
}

ARCH_RESOLUTION = {
    "SimpleCNN": 224,
    "EfficientNetB0": 224,
    "MobileNetV2": 224,
    "Xception": 299,
    "InceptionResNetV2": 299,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_split_df() -> pd.DataFrame:
    return pd.read_excel(SPLIT_XLSX, sheet_name="all")


def load_ground_truth() -> dict:
    with open(GT_JSON) as f:
        data = json.load(f)
    return data["Measurements"]


def dataset_summary_df() -> pd.DataFrame:
    df = load_split_df()
    pivot = df.groupby(["variety", "split"]).size().unstack(fill_value=0)
    for col in ["train", "valid", "test"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot[["train", "valid", "test"]].sum(axis=1)
    pivot = pivot[["train", "valid", "test", "total"]].reset_index()
    total_row = pd.DataFrame([{
        "variety": "Total",
        "train": pivot["train"].sum(),
        "valid": pivot["valid"].sum(),
        "test": pivot["test"].sum(),
        "total": pivot["total"].sum(),
    }])
    return pd.concat([pivot, total_row], ignore_index=True)


def build_manifest(input_type: str, resolution: int) -> pd.DataFrame:
    """Manifest columns: filepath, filename, variety, label_idx, split, + regression targets."""
    split_df = load_split_df()
    gt = load_ground_truth()
    base_name = INPUT_FOLDER_BASENAME[input_type]
    orig_dir = DATASET_ROOT / f"{base_name}_{resolution}"

    id_by_filename = dict(zip(split_df["filename"], split_df["id"]))
    variety_by_filename = dict(zip(split_df["filename"], split_df["variety"]))

    rows = []

    # --- train: from augmentation_log.csv ---
    aug_log = pd.read_csv(AUG_LOG_CSV)
    folder_name = f"{base_name}_{resolution}_aug"
    train_rows = aug_log[aug_log["folder"] == folder_name]
    if len(train_rows) == 0:
        raise FileNotFoundError(f"No augmentation_log rows found for folder={folder_name}")

    for _, r in train_rows.iterrows():
        orig_fname = r["original_filename"]
        image_id = int(id_by_filename[orig_fname])
        gt_row = gt[f"Image{image_id}"]
        targets = {t: gt_row[t] for t in REGRESSION_TARGETS}
        rows.append({
            "filepath": r["output_path"],
            "filename": orig_fname,
            "variety": r["variety"],
            "label_idx": VARIETY_TO_IDX[r["variety"]],
            "split": "train",
            **targets,
        })

    # --- valid/test: from original resized folders ---
    for _, r in split_df.iterrows():
        if r["split"] == "train":
            continue
        filename = r["filename"]
        image_id = int(r["id"])
        gt_row = gt[f"Image{image_id}"]
        targets = {t: gt_row[t] for t in REGRESSION_TARGETS}
        p = orig_dir / filename
        if not p.exists():
            raise FileNotFoundError(f"Missing original file: {p}")
        rows.append({
            "filepath": str(p),
            "filename": filename,
            "variety": r["variety"],
            "label_idx": VARIETY_TO_IDX[r["variety"]],
            "split": r["split"],
            **targets,
        })

    return pd.DataFrame(rows)


def compute_target_norm_stats(train_df: pd.DataFrame) -> dict:
    """Mean/std from de-duplicated (original, non-augmented) train rows only."""
    dedup = train_df.drop_duplicates(subset=["filename"])
    stats = {}
    for t in REGRESSION_TARGETS:
        stats[t] = {"mean": float(dedup[t].mean()), "std": float(dedup[t].std())}
    return stats


def compute_pixel_norm_stats(train_df: pd.DataFrame, input_type: str, resolution: int,
                              sample_n: int = 200) -> dict:
    """Per-channel pixel mean/std for SimpleCNN's own z-score normalization,
    computed from a sample of de-duplicated original training images."""
    dedup = train_df.drop_duplicates(subset=["filename"])
    if len(dedup) > sample_n:
        dedup = dedup.sample(n=sample_n, random_state=42)

    channels = 1 if input_type == "mask" else 3
    sums = np.zeros(channels)
    sqsums = np.zeros(channels)
    n_pixels = 0

    base_name = INPUT_FOLDER_BASENAME[input_type]
    orig_dir = DATASET_ROOT / f"{base_name}_{resolution}"

    for fname in dedup["filename"]:
        p = orig_dir / fname
        img = Image.open(p).convert("L" if channels == 1 else "RGB")
        arr = np.asarray(img, dtype=np.float64) / 255.0
        if channels == 1:
            arr = arr[..., None]
        sums += arr.reshape(-1, channels).sum(axis=0)
        sqsums += (arr.reshape(-1, channels) ** 2).sum(axis=0)
        n_pixels += arr.shape[0] * arr.shape[1]

    mean = sums / n_pixels
    var = sqsums / n_pixels - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))
    return {"mean": mean.tolist(), "std": std.tolist()}


class LettuceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, resolution: int, input_type: str, task: str,
                 pixel_norm: dict | None = None, target_norm: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.resolution = resolution
        self.input_type = input_type
        self.task = task
        self.channels = 1 if input_type == "mask" else 3
        self.pixel_norm = pixel_norm  # {"mean": [...], "std": [...]} in [0,1] scale
        self.target_norm = target_norm  # {trait: {"mean":..,"std":..}}

    def __len__(self):
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L" if self.channels == 1 else "RGB")
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.channels == 1:
            arr = np.repeat(arr[..., None], 3, axis=-1)  # replicate mask to 3 channels
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        if self.pixel_norm is not None:
            # SimpleCNN: z-score with training-set pixel mean/std
            mean = torch.tensor(self.pixel_norm["mean"], dtype=torch.float32)
            std = torch.tensor(self.pixel_norm["std"], dtype=torch.float32)
            if self.channels == 1:
                mean = mean.repeat(3)
                std = std.repeat(3)
            tensor = (tensor - mean[:, None, None]) / std[:, None, None]
        else:
            # Pretrained backbones: ImageNet normalization
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32)
            tensor = (tensor - mean[:, None, None]) / std[:, None, None]

        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row["filepath"])

        if self.task == "classification":
            label = torch.tensor(row["label_idx"], dtype=torch.long)
            return img, label
        else:
            vals = []
            for t in REGRESSION_TARGETS:
                v = row[t]
                if self.target_norm is not None:
                    v = (v - self.target_norm[t]["mean"]) / self.target_norm[t]["std"]
                vals.append(v)
            target = torch.tensor(vals, dtype=torch.float32)
            return img, target
