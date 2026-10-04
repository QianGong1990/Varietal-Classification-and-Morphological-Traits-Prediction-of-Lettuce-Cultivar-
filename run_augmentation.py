import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

ROOT = Path("dataset")
AUG_ROOT = ROOT / "augmented"
RESOLUTIONS = [224, 299]

SPLIT_XLSX = ROOT / "split" / "dataset_split.xlsx"


def adjust_hue(img: Image.Image, delta: float) -> Image.Image:
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    h_arr = np.array(h).astype(np.int16)
    shift = int(round(delta * 255))
    h_arr = (h_arr + shift) % 256
    h_new = Image.fromarray(h_arr.astype(np.uint8), mode="L")
    return Image.merge("HSV", (h_new, s, v)).convert("RGB")


RGB_TRANSFORMS = {
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
assert len(RGB_TRANSFORMS) == 16

MASK_TRANSFORMS = {
    "original": lambda im: im,
    "rot90": lambda im: im.rotate(90, expand=False),
    "rot180": lambda im: im.rotate(180),
    "rot270": lambda im: im.rotate(270, expand=False),
    "hflip": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "vflip": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
}
assert len(MASK_TRANSFORMS) == 6


def rethreshold_mask(im: Image.Image) -> Image.Image:
    arr = np.array(im)
    arr = np.where(arr > 127, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def out_name(stem: str, transform: str) -> str:
    return f"{stem}_{transform}.png"


def main():
    df = pd.read_excel(SPLIT_XLSX, sheet_name="all")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    variety_by_file = dict(zip(train_df["filename"], train_df["variety"]))
    train_files = train_df["filename"].tolist()
    print(f"Training files: {len(train_files)}")

    log_rows = []
    folder_counts = {}

    for res in RESOLUTIONS:
        rgb_src_dir = ROOT / f"RGBImages_{res}"
        seg_src_dir = ROOT / f"removed_background_{res}"
        mask_src_dir = ROOT / f"binary_masks_{res}"

        rgb_dst_dir = AUG_ROOT / f"RGBImages_{res}"
        seg_dst_dir = AUG_ROOT / f"removed_background_{res}"
        mask_dst_dir = AUG_ROOT / f"binary_masks_{res}"
        for d in (rgb_dst_dir, seg_dst_dir, mask_dst_dir):
            d.mkdir(parents=True, exist_ok=True)

        rgb_count = 0
        seg_count = 0
        mask_count = 0

        for fname in train_files:
            stem = Path(fname).stem
            variety = variety_by_file[fname]

            rgb_img = Image.open(rgb_src_dir / fname).convert("RGB")
            seg_img = Image.open(seg_src_dir / fname).convert("RGB")
            mask_img = Image.open(mask_src_dir / fname).convert("L")

            for tname, tfn in RGB_TRANSFORMS.items():
                out_fn = out_name(stem, tname)

                rgb_out = tfn(rgb_img).convert("RGB")
                rgb_out.save(rgb_dst_dir / out_fn)
                rgb_count += 1
                log_rows.append({
                    "original_filename": fname, "variety": variety, "split": "train",
                    "resolution": res, "folder": f"RGBImages_{res}",
                    "transform": tname, "output_filename": out_fn,
                    "output_path": str(rgb_dst_dir / out_fn),
                })

                seg_out = tfn(seg_img).convert("RGB")
                seg_out.save(seg_dst_dir / out_fn)
                seg_count += 1
                log_rows.append({
                    "original_filename": fname, "variety": variety, "split": "train",
                    "resolution": res, "folder": f"removed_background_{res}",
                    "transform": tname, "output_filename": out_fn,
                    "output_path": str(seg_dst_dir / out_fn),
                })

            for tname, tfn in MASK_TRANSFORMS.items():
                out_fn = out_name(stem, tname)
                mask_out = rethreshold_mask(tfn(mask_img))
                mask_out.save(mask_dst_dir / out_fn)
                mask_count += 1
                log_rows.append({
                    "original_filename": fname, "variety": variety, "split": "train",
                    "resolution": res, "folder": f"binary_masks_{res}",
                    "transform": tname, "output_filename": out_fn,
                    "output_path": str(mask_dst_dir / out_fn),
                })

        folder_counts[f"RGBImages_{res}"] = rgb_count
        folder_counts[f"removed_background_{res}"] = seg_count
        folder_counts[f"binary_masks_{res}"] = mask_count
        print(f"Resolution {res}: RGB={rgb_count}, segmented={seg_count}, masks={mask_count}")

    log_df = pd.DataFrame(log_rows)
    log_path = AUG_ROOT / "augmentation_log.csv"
    log_df.to_csv(log_path, index=False)

    print()
    print("=== Final counts per folder ===")
    for folder, count in folder_counts.items():
        actual = len(list((AUG_ROOT / folder).glob("*.png")))
        print(f"{folder}: expected={count}, actual files on disk={actual}")

    print()
    print(f"Augmentation log saved: {log_path} ({len(log_df)} rows)")


if __name__ == "__main__":
    main()
