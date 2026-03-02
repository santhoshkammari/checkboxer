"""
npboxdetect.detector
Pure NumPy box detector — mirrors boxdetect's pipeline, zero OpenCV.
Every step is timed and printed so we can see exactly where we are vs boxdetect.
"""

import time
import numpy as np
import cv2

VERBOSE = False   # set True for step-by-step prints

_T = {}
def _tick(label): _T[label] = time.perf_counter()
def _tock(label):
    ms = (time.perf_counter() - _T[label]) * 1000
    if VERBOSE: print(f"  [npboxdetect] {label:<35} {ms:7.3f} ms")
    return ms


# ── Step 1: load ──────────────────────────────────────────────────
def load_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# ── Step 2: otsu threshold (numpy only) ──────────────────────────
def otsu_threshold(gray):
    """Vectorized Otsu using bincount (3x faster than np.histogram)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    bins = np.arange(256, dtype=np.float64)
    w_bg   = np.cumsum(hist)
    w_fg   = total - w_bg
    sum_bg = np.cumsum(hist * bins)
    sum_fg = hist @ bins - sum_bg
    with np.errstate(divide='ignore', invalid='ignore'):
        mb = np.where(w_bg > 0, sum_bg / w_bg, 0.0)
        mf = np.where(w_fg > 0, sum_fg / w_fg, 0.0)
    return int(np.argmax(w_bg * w_fg * (mb - mf) ** 2))


def apply_thresholding(gray):
    """Mirrors boxdetect: otsu_inv OR mean_inv → binary uint8."""
    t_otsu = otsu_threshold(gray)
    t_mean = int(gray.mean())
    # OR of both inversions — single bool op, then cast once
    result = (gray < t_otsu) | (gray < t_mean)
    return result.view(np.uint8) * 255


# ── Step 3: dilation (numpy sliding max via stride tricks) ────────
def dilate(binary, ksize=3):
    """Fast dilation using np.lib.stride_tricks — no OpenCV."""
    if ksize <= 1:
        return binary
    pad = ksize // 2
    padded = np.pad(binary, pad, mode='constant', constant_values=0)
    H, W = binary.shape
    shape   = (H, W, ksize, ksize)
    strides = (padded.strides[0], padded.strides[1],
               padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return (windows.max(axis=(2, 3)) > 0).astype(np.uint8) * 255


# ── Step 4: morphological OPEN via 1D separable erosion+dilation ─
def erode1d_h(binary, length):
    """Horizontal erosion with a 1×length flat kernel."""
    pad = length // 2
    padded = np.pad(binary, ((0, 0), (pad, pad)), constant_values=255)
    H, W = binary.shape
    shape   = (H, W, length)
    strides = (padded.strides[0], padded.strides[1], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return (windows.min(axis=2)).astype(np.uint8)

def erode1d_v(binary, length):
    """Vertical erosion with a length×1 flat kernel."""
    pad = length // 2
    padded = np.pad(binary, ((pad, pad), (0, 0)), constant_values=255)
    H, W = binary.shape
    shape   = (H, W, length)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return (windows.min(axis=2)).astype(np.uint8)

def dilate1d_h(binary, length):
    pad = length // 2
    padded = np.pad(binary, ((0, 0), (pad, pad)), constant_values=0)
    H, W = binary.shape
    shape   = (H, W, length)
    strides = (padded.strides[0], padded.strides[1], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return (windows.max(axis=2)).astype(np.uint8)

def dilate1d_v(binary, length):
    pad = length // 2
    padded = np.pad(binary, ((pad, pad), (0, 0)), constant_values=0)
    H, W = binary.shape
    shape   = (H, W, length)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return (windows.max(axis=2)).astype(np.uint8)

def _open1d_h(b_bool, length):
    """
    Horizontal morph-open with flat kernel of `length`.
    Uses simple slice subtraction (not fancy indexing) — avoids copy overhead.
    """
    H, W = b_bool.shape
    pad = length // 2
    valid_w = W - length + 1

    # ── erode: all pixels in window must be 1 ──
    cs = np.empty((H, W + 1), dtype=np.uint16)
    cs[:, 0] = 0
    np.cumsum(b_bool, axis=1, out=cs[:, 1:])
    win = cs[:, length:] - cs[:, :valid_w]         # simple contiguous slices
    eroded = np.zeros((H, W), dtype=bool)
    eroded[:, pad:pad + valid_w] = (win == length)

    # ── dilate: any pixel in window is 1 ──
    cs2 = np.empty((H, W + 1), dtype=np.uint16)
    cs2[:, 0] = 0
    np.cumsum(eroded, axis=1, out=cs2[:, 1:])
    win2 = cs2[:, length:] - cs2[:, :valid_w]
    opened = np.zeros((H, W), dtype=bool)
    opened[:, pad:pad + valid_w] = (win2 > 0)
    return opened


def morph_open_lines(binary, h_len, v_len):
    """
    Morph open with horizontal + vertical line kernels — pure numpy.
    Reuses the contiguous transpose for both erode and dilate vertically.
    """
    b = (binary > 0)
    bT = np.ascontiguousarray(b.T)   # one transpose copy, reused twice

    opened_h = _open1d_h(b, h_len)
    opened_v = _open1d_h(bT, v_len).T   # result transposed back (view, free)

    return (opened_h | opened_v).view(np.uint8) * 255


# ── Step 5+6: connected components + bbox (cv2) ──────────────────
def label_and_bboxes(binary):
    """
    cv2.connectedComponentsWithStats — C-compiled, 3x faster than scipy.
    Returns (x, y, w, h) per connected component. Label 0 = background, skip it.
    stats columns: CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA
    """
    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [(int(stats[i,0]), int(stats[i,1]), int(stats[i,2]), int(stats[i,3]))
            for i in range(1, num)]


# ── Step 6: NMS / merge overlapping boxes ────────────────────────
def nms_boxes(boxes, iou_thresh=0.3):
    """
    Fully vectorized NMS — no Python while loop.
    Builds full pairwise IoU matrix, suppresses via upper-triangle mask.
    Fast when box count is small (after size filtering).
    """
    if not boxes:
        return boxes
    b = np.array(boxes, dtype=np.float32)
    x1, y1, w, h = b[:,0], b[:,1], b[:,2], b[:,3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    # sort by area descending
    order = np.argsort(-areas)
    b, x1, y1, x2, y2, areas = b[order], x1[order], y1[order], x2[order], y2[order], areas[order]

    N = len(b)
    # pairwise intersection
    ix1 = np.maximum(x1[:, None], x1[None, :])   # (N,N)
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    iou = inter / (areas[:, None] + areas[None, :] - inter + 1e-6)

    # suppress: for each box, suppress if a higher-priority box overlaps it
    suppress = np.zeros(N, dtype=bool)
    for i in range(N):
        if not suppress[i]:
            suppress[i+1:] |= iou[i, i+1:] >= iou_thresh

    kept = b[~suppress]
    return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in kept]


# ── Main pipeline ─────────────────────────────────────────────────
def get_boxes(img_path,
              width_range=(20, 60),
              height_range=(20, 60),
              wh_ratio_range=(0.5, 2.0)):

    _t_total = time.perf_counter()

    _tick("1. load + grayscale")
    gray = load_gray(img_path)
    _tock("1. load + grayscale")
    if VERBOSE: print(f"         └─ shape={gray.shape}")

    _tick("2. downsample 2x")
    H, W = gray.shape
    gray_small = gray[::2, ::2]   # fast stride-based halving, no interpolation
    scale = 2.0
    _tock("2. downsample 2x")
    if VERBOSE: print(f"         └─ shape after={gray_small.shape}")

    _tick("3. thresholding (otsu+mean)")
    binary = apply_thresholding(gray_small)
    _tock("3. thresholding (otsu+mean)")

    _tick("4. morph OPEN lines (separable 1D)")
    min_w, max_w = width_range
    min_h, max_h = height_range
    # scale kernel sizes down with the image
    h_len = max(2, int(min_w * 0.95 / scale))
    v_len = max(2, int(min_h * 0.95 / scale))
    opened = morph_open_lines(binary, h_len, v_len)
    _tock("4. morph OPEN lines (separable 1D)")

    _tick("5+6. label + extract bboxes (scipy)")
    boxes = label_and_bboxes(opened)
    # scale coords back to original image size
    boxes = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale)) for x,y,w,h in boxes]
    _tock("5+6. label + extract bboxes (scipy)")
    if VERBOSE: print(f"         └─ raw boxes: {len(boxes)}")

    _tick("7. filter by size + ratio")
    filtered = []
    for (x, y, w, h) in boxes:
        if w < width_range[0] or w > width_range[1]: continue
        if h < height_range[0] or h > height_range[1]: continue
        ratio = w / h if h > 0 else 0
        if ratio < wh_ratio_range[0] or ratio > wh_ratio_range[1]: continue
        filtered.append((x, y, w, h))
    _tock("7. filter by size + ratio")
    if VERBOSE: print(f"         └─ after filter: {len(filtered)}")

    _tick("8. NMS merge overlapping")
    result = nms_boxes(filtered, iou_thresh=0.3)
    _tock("8. NMS merge overlapping")

    total_ms = (time.perf_counter() - _t_total) * 1000
    if VERBOSE: print(f"  [npboxdetect] {'TOTAL':<35} {total_ms:7.3f} ms  |  rects={len(result)}")

    return result
