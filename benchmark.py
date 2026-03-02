"""
Benchmark: checkbox detection approaches on data/ images
Approaches:
  1. npboxdetect  — our pure-numpy+numba library
  2. boxdetect    — reference OpenCV pipeline
  3. opencv_contours
  4. morphology
"""

import time
import glob
import os
import cv2
import numpy as np

DATA_IMAGES = sorted(glob.glob("data/*.png"))
RUNS = 50


# ── Approach 1: npboxdetect ───────────────────────────────────────────────────
import npboxdetect  # triggers numba warmup at import
from npboxdetect.detector import get_boxes as _np_get_boxes

def approach_npboxdetect(img_path):
    return _np_get_boxes(img_path)


# ── Approach 2: boxdetect ─────────────────────────────────────────────────────
from boxdetect.pipelines import get_boxes as _bd_get_boxes
from boxdetect.config import PipelinesConfig

_bd_cfg = PipelinesConfig()
_bd_cfg.width_range = (20, 60)
_bd_cfg.height_range = (20, 60)
_bd_cfg.scaling_factors = [1.0]
_bd_cfg.wh_ratio_range = (0.5, 2.0)
_bd_cfg.group_size_range = (1, 100)
_bd_cfg.dilation_iterations = 0

def approach_boxdetect(img_path):
    img = cv2.imread(os.path.abspath(img_path))
    rects = _bd_get_boxes(img, _bd_cfg, plot=False)[0]
    return [(x, y, w, h) for (x, y, w, h) in rects] if rects is not None else []


# ── Approach 3: OpenCV contours ───────────────────────────────────────────────
def approach_opencv_contours(img_path):
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


# ── Approach 4: Morphology + contours ────────────────────────────────────────
def approach_morphology(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / h if h > 0 else 0
        if 10 < w < 80 and 10 < h < 80 and 0.5 < ratio < 2.0 and w * h > 100:
            boxes.append((x, y, w, h))
    return boxes


# ── Draw & save result image ──────────────────────────────────────────────────
def save_result_image(img_path, approach_name, bboxes):
    img = cv2.imread(os.path.abspath(img_path))
    for (x, y, w, h) in bboxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    os.makedirs("results", exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(f"results/{base}__{approach_name}.png", img)


# ── Runner ────────────────────────────────────────────────────────────────────
approaches = {
    "npboxdetect":     approach_npboxdetect,
    "boxdetect":       approach_boxdetect,
    "opencv_contours": approach_opencv_contours,
    "morphology":      approach_morphology,
}

print(f"\n{'='*72}")
print(f"  CHECKBOX DETECTION BENCHMARK  |  images: {len(DATA_IMAGES)}  |  runs: {RUNS}")
print(f"{'='*72}\n")

all_results = {}

for img_path in DATA_IMAGES:
    print(f"Image: {img_path}")
    print(f"  {'Approach':<20} {'p50 (ms)':>10} {'p10 (ms)':>10} {'p90 (ms)':>10} {'boxes':>7}")
    print(f"  {'-'*60}")

    for name, fn in approaches.items():
        times = []
        bboxes = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            try:
                bboxes = fn(img_path)
            except Exception as e:
                print(f"  {name:<20} ERROR: {e}")
                break
            times.append((time.perf_counter() - t0) * 1000)

        if times:
            times.sort()
            p10 = times[max(0, RUNS // 10)]
            p50 = times[RUNS // 2]
            p90 = times[min(RUNS - 1, RUNS * 9 // 10)]
            print(f"  {name:<20} {p50:>10.1f} {p10:>10.1f} {p90:>10.1f} {len(bboxes):>7}")
            all_results.setdefault(name, {})[img_path] = {"p50": p50, "bboxes": bboxes}
            save_result_image(img_path, name, bboxes)

    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  SUMMARY — p50 across all images (ms)")
print(f"{'='*72}")
print(f"  {'Approach':<20} {'avg p50':>10}   vs npboxdetect")
np_vals = [v["p50"] for v in all_results.get("npboxdetect", {}).values()]
np_avg = np.mean(np_vals) if np_vals else 1.0
for name in approaches:
    vals = [v["p50"] for v in all_results.get(name, {}).values()]
    avg = np.mean(vals) if vals else float("nan")
    ratio = avg / np_avg if np_avg else float("nan")
    marker = " <-- baseline" if name == "npboxdetect" else f" ({ratio:.1f}x slower)"
    print(f"  {name:<20} {avg:>10.1f} ms{marker}")
print()
