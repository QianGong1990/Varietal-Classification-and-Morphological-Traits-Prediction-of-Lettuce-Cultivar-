import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
from scipy import ndimage

INPUT_DIR = Path("dataset/RGBImages")
OUTPUT_DIR = Path("dataset/removed_backgound")


def crate_bbox(rgb: np.ndarray):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h >= 8) & (h <= 25) & (s > 90) & (v > 80)).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    # opening first drops thin spurious bridges (reflections/edges) that would
    # otherwise fuse an ear's bbox with unrelated stuff via a hairline connection
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    labeled, n = ndimage.label(mask > 0)
    if n == 0:
        return None
    sizes = ndimage.sum(mask > 0, labeled, range(1, n + 1))
    max_size = sizes.max()
    if max_size < 1000:
        return None

    boxes = []
    for lab in range(1, n + 1):
        if sizes[lab - 1] < max(0.15 * max_size, 1000):
            continue
        ys, xs = np.where(labeled == lab)
        boxes.append((sizes[lab - 1], xs.min(), ys.min(), xs.max(), ys.max()))
    if not boxes:
        return None

    # The real crate's ears are a fairly narrow, consistent shape (~150-380px
    # wide). Other orange props in frame (e.g. a stack of spare crates) tend
    # to be wider/taller blobs, so use the ear-shaped candidates to anchor on
    # the real crate instead of just taking the largest orange blob overall.
    def is_ear_shaped(box):
        _, bx0, by0, bx1, by1 = box
        w, h_ = bx1 - bx0, by1 - by0
        return 150 <= w <= 300 and 300 <= h_ <= 660

    ear_like = [b for b in boxes if is_ear_shaped(b)]
    anchor_pool = ear_like if ear_like else boxes

    # The real crate is made of two ears of similar size, vertically aligned
    # (same y-extent), sitting left/right of each other. Other orange props
    # elsewhere in the greenhouse (e.g. stacked spare crates) are typically a
    # different size and/or not aligned with the anchor, so they get skipped.
    anchor_pool.sort(reverse=True)
    anchor_size, ax0, ay0, ax1, ay1 = anchor_pool[0]
    merged = [ax0, ay0, ax1, ay1]

    best = None
    for box in boxes:
        size, bx0, by0, bx1, by1 = box
        if (bx0, by0, bx1, by1) == (ax0, ay0, ax1, ay1):
            continue
        if not is_ear_shaped(box):
            continue
        if not (0.2 * anchor_size <= size <= 2.8 * anchor_size):
            continue
        y_overlap = max(0, min(ay1, by1) - max(ay0, by0))
        y_overlap_frac = y_overlap / max(min(ay1 - ay0, by1 - by0), 1)
        if y_overlap_frac < 0.5:
            continue
        x_overlap = max(0, min(ax1, bx1) - max(ax0, bx0))
        if x_overlap > 0.3 * min(ax1 - ax0, bx1 - bx0):
            continue  # should be side-by-side, not overlapping in x
        if best is None or size > best[0]:
            best = (size, bx0, by0, bx1, by1)

    if best is not None:
        _, bx0, by0, bx1, by1 = best
        merged[0] = min(merged[0], bx0)
        merged[1] = min(merged[1], by0)
        merged[2] = max(merged[2], bx1)
        merged[3] = max(merged[3], by1)
    else:
        # No matching ear found -- often because the other ear's mask got
        # bridged (via ambiguous plant/shadow color) into a too-wide blob
        # that fails the shape filter and gets dropped entirely. The crate's
        # total width is very consistent across this rig (~880-950px), so
        # extend the lone anchor out to that nominal width in whichever
        # direction the missing ear should be, rather than leaving the crop
        # confined to a single ear (which would cut the plant in half).
        nominal_width = 900
        img_w = rgb.shape[1]
        if ax0 > img_w - ax1:  # anchor sits on the right half -> missing ear is to the left
            merged[0] = max(ax1 - nominal_width, 0)
        else:  # anchor sits on the left half -> missing ear is to the right
            merged[2] = min(ax0 + nominal_width, img_w - 1)

    return tuple(merged)


def find_tray_bbox(rgb: np.ndarray, crate_box):
    x0, y0, x1, y1 = crate_box
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (15, 15))
    sqm = cv2.blur(gray * gray, (15, 15))
    std = np.sqrt(np.clip(sqm - mean * mean, 0, None))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # white/light tray surface, visible when the plant doesn't fully cover it
    tray_mask = (s < 50) & (v > 150) & (std < 10)

    # vegetation: green/yellow/olive/brown (excludes bright orange crate hue),
    # plus dark red/purple leaf varieties. The crate's orange hue can extend
    # into shadowed grooves that read as similarly dark, but empirically the
    # crate's shadow value stays well above true dark-leaf value (median ~118
    # vs ~50), so a tight value cutoff separates them without excluding a
    # whole dilated crate region (which would erase dark-leaf plants too).
    veg_mask = ((h >= 26) & (h <= 100) & (s >= 40) & (v >= 20)) | (
        (h < 26) & (s >= 60) & (v < 90)
    )

    kernel_small = np.ones((5, 5), np.uint8)

    def find_candidates(bool_mask):
        m = bool_mask.astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        roi = np.zeros_like(m)
        roi[y0:y1, x0:x1] = m[y0:y1, x0:x1]
        labeled, n = ndimage.label(roi > 0)
        out = []
        if n == 0:
            return out
        sizes = ndimage.sum(roi > 0, labeled, range(1, n + 1))
        for lab in range(1, n + 1):
            if sizes[lab - 1] < 800:
                continue
            ys, xs = np.where(labeled == lab)
            bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
            w, h_ = bx1 - bx0 + 1, by1 - by0 + 1
            aspect = max(w, h_) / max(min(w, h_), 1)
            if aspect > 4.5:
                continue
            out.append((sizes[lab - 1], bx0, by0, bx1, by1))
        return out

    veg_candidates = find_candidates(veg_mask)
    tray_candidates = find_candidates(tray_mask)
    candidates = veg_candidates + tray_candidates

    if not candidates:
        return crate_box

    # Prefer anchoring on vegetation: several white trays can be visible at
    # once (e.g. spare trays stacked near the one holding the plant), and
    # anchoring on "the biggest bright blob" would grab an empty tray instead
    # of the one with the plant. Vegetation color is specific to the plant.
    anchor_pool = veg_candidates if veg_candidates else candidates
    anchor_pool = sorted(anchor_pool, reverse=True)
    primary = anchor_pool[0]
    merged = list(primary[1:])
    remaining = [c for c in candidates if c != primary]

    # iteratively absorb boxes that are physically adjacent/overlapping the
    # growing merged region (not just aligned along one axis far away). Capped
    # so that, e.g., a stack of several spare trays next to the one holding
    # the plant doesn't all get swept into one enormous crop -- that confuses
    # rembg's second-pass saliency into picking the wrong object entirely.
    gap_tol = 40
    max_w, max_h = 700, 600
    changed = True
    while changed:
        changed = False
        still_remaining = []
        for size, bx0, by0, bx1, by1 in remaining:
            dx = max(0, max(merged[0], bx0) - min(merged[2], bx1))
            dy = max(0, max(merged[1], by0) - min(merged[3], by1))
            new_x0, new_y0 = min(merged[0], bx0), min(merged[1], by0)
            new_x1, new_y1 = max(merged[2], bx1), max(merged[3], by1)
            if (
                dx <= gap_tol
                and dy <= gap_tol
                and (new_x1 - new_x0) <= max_w
                and (new_y1 - new_y0) <= max_h
            ):
                merged[0], merged[1], merged[2], merged[3] = new_x0, new_y0, new_x1, new_y1
                changed = True
            else:
                still_remaining.append((size, bx0, by0, bx1, by1))
        remaining = still_remaining

    pad = 60
    mx0, my0, mx1, my1 = merged
    mx0 = max(mx0 - pad, 0)
    my0 = max(my0 - pad, 0)
    mx1 = min(mx1 + pad, rgb.shape[1] - 1)
    my1 = min(my1 + pad, rgb.shape[0] - 1)
    return mx0, my0, mx1, my1


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask > 0)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask > 0, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    return np.where(labeled == largest, 255, 0).astype(np.uint8)


def most_plant_like_component(mask: np.ndarray, veg_mask: np.ndarray) -> np.ndarray:
    """Pick the largest component that plausibly contains real plant material,
    not simply the largest overall -- rembg's raw foreground selection can
    include a second, unrelated region (background shadow, another tray)
    alongside the real plant, sometimes bigger than the plant itself. A pure
    "most vegetation pixels" pick is *also* wrong: a partly-desaturated dying
    leaf can be mostly outside the vegetation color range yet still be the
    right answer, while a tiny fully-saturated scrap elsewhere would win on
    raw overlap. So: require a modest vegetation fraction to qualify at all
    (weeds out non-plant blobs), then take the largest qualifier by size.
    """
    labeled, n = ndimage.label(mask > 0)
    if n == 0:
        return mask
    if n == 1:
        return np.where(labeled == 1, 255, 0).astype(np.uint8)
    sizes = ndimage.sum(mask > 0, labeled, range(1, n + 1))
    veg_overlap = ndimage.sum(veg_mask, labeled, range(1, n + 1))
    veg_frac = veg_overlap / np.maximum(sizes, 1)

    qualifies = veg_frac >= 0.05
    if not qualifies.any():
        return largest_component(mask)
    candidate_labels = np.where(qualifies)[0] + 1
    best = candidate_labels[np.argmax(sizes[candidate_labels - 1])]
    return np.where(labeled == best, 255, 0).astype(np.uint8)


def process_image(session, src_path: Path, dst_path: Path) -> None:
    rgb_img = Image.open(src_path).convert("RGB")
    rgb = np.array(rgb_img)

    cbox = crate_bbox(rgb)
    if cbox is None:
        cbox = (0, 0, rgb.shape[1] - 1, rgb.shape[0] - 1)

    tx0, ty0, tx1, ty1 = find_tray_bbox(rgb, cbox)

    crop = rgb_img.crop((tx0, ty0, tx1, ty1))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    out = remove(buf.getvalue(), session=session)
    cutout = Image.open(io.BytesIO(out)).convert("RGBA")
    alpha = np.array(cutout)[..., 3]

    crop_rgb = np.array(crop)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grey_ish = (s < 45) & (v > 50)
    veg_mask = ((h >= 26) & (h <= 100) & (s >= 40) & (v >= 20)) | (
        (h < 26) & (s >= 60) & (v < 90)
    )

    # grey/white pixels (tray, metal, clips) must be dropped BEFORE picking the
    # largest component -- if rembg's foreground blob happens to be a bright
    # tray rather than the plant (common when multiple trays are in frame),
    # selecting largest-component first would lock onto the tray and leave
    # nothing once grey pixels are then zeroed out of it.
    alpha_fg = alpha > 30
    alpha_mask = alpha_fg.astype(np.uint8) * 255
    alpha_mask[grey_ish] = 0
    alpha_mask = most_plant_like_component(alpha_mask, veg_mask)

    # rembg's saliency occasionally locks onto a bright tray instead of the
    # plant, or only recovers a small fragment of it (more likely when
    # several trays, or a stray green object like a plastic ID tag, are
    # visible in the crop). A pure color-mask reconstruction is the
    # alternative candidate; comparing which one is bigger (among candidates
    # that are actually plant-colored) picks correctly in both directions:
    # it doesn't let an unrelated green tag's mask outcompete a real but
    # only-partly-saturated plant (tag mask stays smaller), and it doesn't
    # let a tiny rembg fragment win over a color mask that recovers the
    # whole plant.
    component_size = (alpha_mask > 0).sum()
    own_overlap = ((alpha_mask > 0) & veg_mask).sum() if component_size else 0
    own_frac = own_overlap / component_size if component_size else 0

    if veg_mask.sum() > 500:
        fallback = veg_mask.astype(np.uint8) * 255
        fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        fallback[grey_ish] = 0
        fallback = largest_component(fallback)
        fallback_size = (fallback > 0).sum()
        fallback_overlap = ((fallback > 0) & veg_mask).sum() if fallback_size else 0
        fallback_frac = fallback_overlap / fallback_size if fallback_size else 0

        # own_frac >= 0.05 mirrors most_plant_like_component's own qualifying
        # bar: below that, its "primary" pick wasn't chosen for being
        # plant-colored at all (nothing qualified, so it fell back to plain
        # largest-component), meaning size alone says nothing about whether
        # it's trustworthy. At or above that bar, trust its shape and only
        # override with the color mask if that mask is *both* plant-colored
        # and bigger (rembg missed part of a real plant).
        primary_trustworthy = own_frac >= 0.05
        fallback_ok = fallback_frac >= 0.15
        if fallback_ok and (not primary_trustworthy or fallback_size > component_size):
            alpha_mask = fallback

    kernel = np.ones((3, 3), np.uint8)
    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    alpha_mask = most_plant_like_component(alpha_mask, veg_mask)
    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # fill small internal gaps (e.g. a bright specular highlight on a leaf
    # getting classified as grey and punched out) -- safe because it only
    # closes holes fully enclosed by the mask, never grows the outer boundary
    alpha_mask = ndimage.binary_fill_holes(alpha_mask > 0).astype(np.uint8) * 255
    alpha_mask = cv2.GaussianBlur(alpha_mask, (5, 5), 0)

    a = (alpha_mask.astype(np.float32) / 255.0)[..., None]
    plant_crop = (crop_rgb.astype(np.float32) * a).astype(np.uint8)

    full = np.zeros_like(rgb)
    full[ty0:ty1, tx0:tx1] = plant_crop

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full).save(dst_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(
        INPUT_DIR.glob("*.png"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )
    if not images:
        print(f"No images found in {INPUT_DIR}")
        return

    session = new_session("u2net")

    for i, src_path in enumerate(images, 1):
        dst_path = OUTPUT_DIR / src_path.name
        try:
            process_image(session, src_path, dst_path)
        except Exception as e:
            print(f"[{i}/{len(images)}] FAILED {src_path.name}: {e}")
            continue
        print(f"[{i}/{len(images)}] {src_path.name} -> {dst_path}")

    print(f"\nDone. {len(images)} images written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
