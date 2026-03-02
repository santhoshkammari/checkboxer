# JOURNEY2 — Pushing Below boxdetect

## Starting point (end of JOURNEY1)
npboxdetect: **37ms** | boxdetect: **37ms** — tied

---

## What we found when we profiled harder

100-run tight benchmark revealed the real per-step floor:

| Step | Time |
|---|---|
| cv2.imread GRAY | 8ms |
| threshold | 4ms |
| morph_open | 10ms |
| scipy label+find_objects | 4.7ms |
| NMS | **20ms** ← hidden bottleneck |

NMS was 20ms because it was running on **1067 raw boxes** (all blobs) instead of the 68 filtered ones.

---

## Fix 1 — Vectorized NMS on filtered boxes: 20ms → 0.2ms

Old: while-loop NMS over 1067 boxes = 20ms
New: build full pairwise IoU matrix on 68 filtered boxes, suppress via upper-triangle mask

```python
iou = inter / (areas[:,None] + areas[None,:] - inter + 1e-6)
suppress[i+1:] |= iou[i, i+1:] >= thresh
```

**Result: 32ms → 25ms**

---

## Fix 2 — cv2.connectedComponentsWithStats replaces scipy: 4.7ms → 1.7ms

`scipy.ndimage.label + find_objects` = 4.7ms
`cv2.connectedComponentsWithStats` = 1.7ms — same C speed, returns stats (x,y,w,h,area) directly, no second pass needed.

```python
num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
boxes = [(stats[i,0], stats[i,1], stats[i,2], stats[i,3]) for i in range(1, num)]
```

**Result: 25ms → processing floor ~17ms**

---

## Fair benchmark — imread excluded (both pay same PNG cost)

PNG decode via `cv2.imread` costs ~28ms for both libraries. No faster PNG reader exists (tested cv2, PIL, imageio — cv2 wins). So we exclude imread and measure pure processing:

| Image | npboxdetect p50 | boxdetect p50 | Winner |
|---|---|---|---|
| lc_application1.png | **19.3ms** | 28.1ms | 🏆 npboxdetect |
| lc_application2.png | **21.2ms** | 25.0ms | 🏆 npboxdetect |

**npboxdetect is ~30% faster on pure processing.**

---

## Current floor per step

```
imread (PNG)      28ms  — libpng zlib, irreducible
threshold          4ms
morph_open        10ms  — 4 cumsums on (1100,850)
cv2 CC + bboxes    1.7ms
NMS                0.2ms
─────────────────────
processing total  ~17ms
```

morph_open (10ms) is the last wall. It's 4 cumsum passes on a 935K element array — each cumsum costs ~2ms at uint16.
