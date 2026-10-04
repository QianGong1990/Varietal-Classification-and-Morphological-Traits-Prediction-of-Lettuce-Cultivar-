from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

RGB_SRC = Path("dataset/RGBImages_224/RGB_1.png")
MASK_SRC = Path("dataset/binary_masks_224/RGB_1.png")
OUT_DIR = Path("dataset/augmented/preview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# clear old preview files
for p in OUT_DIR.glob("*.png"):
    p.unlink()


def adjust_hue(img: Image.Image, delta: float) -> Image.Image:
    # TF-style: delta in [-1, 1] maps to a shift of delta * 180 degrees
    # on the hue wheel. PIL's HSV H channel is 0-255 == normalized [0,1)*255.
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    h_arr = np.array(h).astype(np.int16)
    shift = int(round(delta * 255))
    h_arr = (h_arr + shift) % 256
    h_new = Image.fromarray(h_arr.astype(np.uint8), mode="L")
    out = Image.merge("HSV", (h_new, s, v)).convert("RGB")
    return out


# 16 RGB/segmented transforms (manuscript-exact)
RGB_TRANSFORMS = {
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

# 6 mask transforms (geometric only, manuscript-exact)
MASK_TRANSFORMS = {
    "rot90": lambda im: im.rotate(90, expand=False),
    "rot180": lambda im: im.rotate(180),
    "rot270": lambda im: im.rotate(270, expand=False),
    "hflip": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "vflip": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
}


def rethreshold_mask(im: Image.Image) -> Image.Image:
    arr = np.array(im)
    arr = np.where(arr > 127, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main():
    rgb = Image.open(RGB_SRC).convert("RGB")
    mask = Image.open(MASK_SRC).convert("L")

    rgb.save(OUT_DIR / "RGB_1_rgb_original.png")
    mask.save(OUT_DIR / "RGB_1_mask_original.png")

    print("=== RGB/segmented transforms (16 total incl. original) ===")
    print("  original -> saved")
    for name, fn in RGB_TRANSFORMS.items():
        out = fn(rgb).convert("RGB")
        out_path = OUT_DIR / f"RGB_1_rgb_{name}.png"
        out.save(out_path)
        print(f"  {name} -> {out_path}")

    print()
    print("=== Binary mask transforms (6 total incl. original) ===")
    print("  original -> saved")
    for name, fn in MASK_TRANSFORMS.items():
        out = rethreshold_mask(fn(mask))
        out_path = OUT_DIR / f"RGB_1_mask_{name}.png"
        out.save(out_path)
        print(f"  {name} -> {out_path}")

    print()
    print(f"Preview saved to: {OUT_DIR}")
    print(f"Total files: {len(list(OUT_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
