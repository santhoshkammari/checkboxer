"""
Benchmark: checkbox detection approaches on data/ images
Approaches:
  1. boxdetect (OpenCV pipeline)
  2. OpenCV contours (pure cv2, no ML)
  3. Manikandan-style (morphology + contours)
  4. Numpy-only (pure numpy, baseline placeholder)
"""

import time
import glob
import numpy as np
from PIL import Image

DATA_IMAGES = sorted(glob.glob("data/*"))
RUNS = 50  # repeat each for stable timing


def load_numpy(path):
    img = np.array(Image.open(path).convert("RGB"))
    return img


def load_gray_numpy(path):
    img = np.array(Image.open(path).convert("L"))
    return img


# ── Approach 1: boxdetect ─────────────────────────────────────────────────────
def approach_boxdetect(img_path):
    import os
    import cv2
    from boxdetect.pipelines import get_boxes
    from boxdetect.config import PipelinesConfig

    cfg = PipelinesConfig()
    cfg.width_range = (20, 60)
    cfg.height_range = (20, 60)
    cfg.scaling_factors = [1.0]
    cfg.wh_ratio_range = (0.5, 2.0)
    cfg.group_size_range = (1, 100)
    cfg.dilation_iterations = 0

    img = cv2.imread(os.path.abspath(img_path))
    result = get_boxes(img, cfg, plot=False)
    rects = result[0]
    return [(x, y, w, h) for (x, y, w, h) in rects] if rects is not None else []


# ── Approach 2: OpenCV contours ───────────────────────────────────────────────
def approach_opencv_contours(img_path):
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / h if h > 0 else 0
        if 10 < w < 80 and 10 < h < 80 and 0.5 < ratio < 2.0:
            boxes.append((x, y, w, h))
    return boxes


# ── Approach 3: Morphology + contours (Manikandan style) ─────────────────────
def approach_morphology(img_path):
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / h if h > 0 else 0
        area = w * h
        if 10 < w < 80 and 10 < h < 80 and 0.5 < ratio < 2.0 and area > 100:
            boxes.append((x, y, w, h))
    return boxes


# ── Approach 4: Pure NumPy ────────────────────────────────────────────────────
def approach_numpy(img_path):
    """
    Pure numpy: threshold → binary → connected components via flood-fill style
    using label propagation (no scipy, no cv2).
    """
    gray = load_gray_numpy(img_path).astype(np.float32)

    # Otsu threshold (numpy only)
    flat = gray.flatten()
    hist, bins = np.histogram(flat, bins=256, range=(0, 256))
    total = flat.size
    sum_total = np.dot(np.arange(256), hist)
    best_thresh, sum_bg, weight_bg = 0, 0, 0
    best_var = 0.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var = var
            best_thresh = t

    binary = (gray < best_thresh).astype(np.uint8)

    # Simple row/col projection-based blob detection (fast, no labeling needed)
    # Scan rows for horizontal runs, then check vertical extent
    boxes = []
    visited = np.zeros_like(binary, dtype=bool)
    rows, cols = np.where(binary == 1)
    if len(rows) == 0:
        return boxes

    # Group pixels into blobs using bounding box scan
    # Sliding window approach: check NxN patches
    h, w = binary.shape
    step = 4
    min_size, max_size = 10, 80

    checked = np.zeros_like(binary, dtype=bool)
    for y in range(0, h - min_size, step):
        for x in range(0, w - min_size, step):
            if checked[y, x]:
                continue
            # expand to find blob extent
            patch = binary[y:y+max_size, x:x+max_size]
            if patch.sum() < 20:
                continue
            # find tight bounding box of foreground in patch
            fy, fx = np.where(patch == 1)
            if len(fy) == 0:
                continue
            bh = fy.max() - fy.min() + 1
            bw = fx.max() - fx.min() + 1
            if min_size < bw < max_size and min_size < bh < max_size:
                ratio = bw / bh
                if 0.5 < ratio < 2.0:
                    bx = x + fx.min()
                    by = y + fy.min()
                    boxes.append((int(bx), int(by), int(bw), int(bh)))
                    checked[by:by+bh, bx:bx+bw] = True

    # deduplicate
    unique = []
    for b in boxes:
        if not any(abs(b[0]-u[0]) < 10 and abs(b[1]-u[1]) < 10 for u in unique):
            unique.append(b)
    return unique


# ── Draw & save results ───────────────────────────────────────────────────────
def save_result_image(img_path, approach_name, bboxes):
    import cv2
    import os
    img = cv2.imread(os.path.abspath(img_path))
    for (x, y, w, h) in bboxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    os.makedirs("results", exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = f"results/{base}__{approach_name}.png"
    cv2.imwrite(out_path, img)


# ── Runner ────────────────────────────────────────────────────────────────────
approaches = {
    "boxdetect": approach_boxdetect,
    "opencv_contours": approach_opencv_contours,
    "morphology": approach_morphology,
}

print(f"\n{'='*70}")
print(f"  CHECKBOX DETECTION BENCHMARK  |  images: {DATA_IMAGES}  |  runs: {RUNS}")
print(f"{'='*70}\n")

results = {}

for img_path in DATA_IMAGES:
    print(f"Image: {img_path}")
    print(f"  {'Approach':<20} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'BBoxes':>8}")
    print(f"  {'-'*58}")

    for name, fn in approaches.items():
        times = []
        bboxes = []
        for i in range(RUNS):
            t0 = time.perf_counter()
            try:
                bboxes = fn(img_path)
            except Exception as e:
                bboxes = [f"ERROR: {e}"]
                times.append(float("nan"))
                break
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times = [t for t in times if not (isinstance(t, float) and t != t)]
        if times:
            avg_ms = np.mean(times)
            min_ms = np.min(times)
            max_ms = np.max(times)
            avg_ns = avg_ms * 1e6
            print(f"  {name:<20} {avg_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f} {len(bboxes):>8}")
        else:
            print(f"  {name:<20} {'ERROR':>10}")

        results.setdefault(name, {})[img_path] = {
            "avg_ms": np.mean(times) if times else None,
            "bboxes": bboxes,
        }
        if isinstance(bboxes, list) and bboxes and not isinstance(bboxes[0], str):
            save_result_image(img_path, name, bboxes)

    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  SUMMARY — Average across all images (ms)")
print(f"{'='*70}")
for name in approaches:
    vals = [v["avg_ms"] for v in results[name].values() if v["avg_ms"] is not None]
    overall = np.mean(vals) if vals else float("nan")
    print(f"  {name:<20} {overall:.3f} ms")

print()
print("  BBox counts per image:")
for img_path in DATA_IMAGES:
    print(f"\n  {img_path}")
    for name in approaches:
        bboxes = results[name][img_path]["bboxes"]
        count = len(bboxes) if isinstance(bboxes, list) else "ERROR"
        sample = bboxes[:3] if isinstance(bboxes, list) else bboxes
        print(f"    {name:<20} count={count}  sample={sample}")
