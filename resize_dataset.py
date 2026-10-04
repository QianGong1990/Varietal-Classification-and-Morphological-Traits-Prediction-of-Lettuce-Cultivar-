from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("dataset")
CROP = 1080
TARGETS = [224, 299]

SRC_RGB = ROOT / "RGBImages"
SRC_SEG = ROOT / "removed_backgound"
SRC_MASK = ROOT / "binary_masks"

DEST_NAME = {
    "RGBImages": "RGBImages",
    "removed_backgound": "removed_background",
    "binary_masks": "binary_masks",
}


def crop_box_from_centroid(cx: int, cy: int, w: int, h: int, size: int):
    half = size // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + size
    y1 = y0 + size

    clamped = False
    if x0 < 0:
        x1 -= x0
        x0 = 0
        clamped = True
    if y0 < 0:
        y1 -= y0
        y0 = 0
        clamped = True
    if x1 > w:
        x0 -= (x1 - w)
        x1 = w
        clamped = True
    if y1 > h:
        y0 -= (y1 - h)
        y1 = h
        clamped = True

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    return x0, y0, x1, y1, clamped


def main():
    seg_files = sorted(
        SRC_SEG.glob("*.png"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )

    dst_dirs = {}
    for folder in ["RGBImages", "removed_backgound", "binary_masks"]:
        for t in TARGETS:
            d = ROOT / f"{DEST_NAME[folder]}_{t}"
            d.mkdir(parents=True, exist_ok=True)
            dst_dirs[(folder, t)] = d

    processed = 0
    clamped_list = []

    for seg_path in seg_files:
        name = seg_path.name
        rgb_path = SRC_RGB / name
        mask_path = SRC_MASK / name

        if not rgb_path.exists() or not mask_path.exists():
            print(f"SKIP {name}: missing counterpart file")
            continue

        seg_img = Image.open(seg_path).convert("RGB")
        seg_arr = np.array(seg_img)
        w, h = seg_img.size

        non_black = np.any(seg_arr > 10, axis=-1)
        ys, xs = np.where(non_black)
        if len(xs) == 0:
            cx, cy = w // 2, h // 2
        else:
            cx, cy = int(round(xs.mean())), int(round(ys.mean()))

        x0, y0, x1, y1, clamped = crop_box_from_centroid(cx, cy, w, h, CROP)
        if clamped:
            clamped_list.append(name)

        box = (x0, y0, x1, y1)

        rgb_crop = Image.open(rgb_path).convert("RGB").crop(box)
        seg_crop = seg_img.crop(box)
        mask_crop = Image.open(mask_path).convert("L").crop(box)

        for t in TARGETS:
            rgb_r = rgb_crop.resize((t, t), Image.BILINEAR).convert("RGB")
            rgb_r.save(dst_dirs[("RGBImages", t)] / name)

            seg_r = seg_crop.resize((t, t), Image.BILINEAR).convert("RGB")
            seg_r.save(dst_dirs[("removed_backgound", t)] / name)

            mask_r = mask_crop.resize((t, t), Image.NEAREST)
            mask_arr = np.array(mask_r)
            mask_arr = np.where(mask_arr > 127, 255, 0).astype(np.uint8)
            Image.fromarray(mask_arr, mode="L").save(dst_dirs[("binary_masks", t)] / name)

        processed += 1

    print()
    print(f"Processed: {processed} images")
    print(f"Clamped to boundary: {len(clamped_list)} images")
    if clamped_list:
        print("Clamped images:", ", ".join(clamped_list))

    for folder in ["RGBImages", "removed_backgound", "binary_masks"]:
        for t in TARGETS:
            n = len(list(dst_dirs[(folder, t)].glob("*.png")))
            print(f"{DEST_NAME[folder]}_{t}: {n} images saved -> {dst_dirs[(folder, t)]}")


if __name__ == "__main__":
    main()
