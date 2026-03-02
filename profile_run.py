"""
profile_run.py  —  single image deep profiler
Usage: python profile_run.py data/lc_application1.png
Shows per-step timing for boxdetect AND numpydetect side by side.
"""
import sys
import os
import time
import cv2
import numpy as np

img_path = sys.argv[1] if len(sys.argv) > 1 else "data/lc_application1.png"
assert os.path.exists(img_path), f"File not found: {img_path}"

print(f"\n{'='*60}")
print(f"  IMAGE: {img_path}")
print(f"{'='*60}")

# ── boxdetect (instrumented) ──────────────────────────────────────
print("\n[BOXDETECT] step-by-step breakdown:")
print("-" * 60)
from boxdetect.pipelines import get_boxes
from boxdetect.config import PipelinesConfig
cfg = PipelinesConfig()
cfg.width_range = (20, 60)
cfg.height_range = (20, 60)
cfg.scaling_factors = [1.0]
cfg.wh_ratio_range = (0.5, 2.0)
cfg.group_size_range = (1, 100)
cfg.dilation_iterations = 0
img_cv = cv2.imread(os.path.abspath(img_path))
result = get_boxes(img_cv, cfg, plot=False)
bd_rects = result[0]

# ── numpydetect ───────────────────────────────────────────────────
print("\n[NUMPYDETECT] step-by-step breakdown:")
print("-" * 60)
from npboxdetect.detector import get_boxes as np_get_boxes
nd_rects = np_get_boxes(img_path)

# ── summary ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  boxdetect   rects: {len(bd_rects)}")
print(f"  numpydetect rects: {len(nd_rects)}")
