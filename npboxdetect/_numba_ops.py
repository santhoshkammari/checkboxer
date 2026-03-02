import numpy as np
from numba import njit, prange, set_num_threads
import os

# pin to physical cores to reduce thread scheduling jitter
_CORES = max(1, os.cpu_count() // 2)
set_num_threads(_CORES)


@njit(cache=True)
def otsu_threshold(gray):
    """Compute Otsu threshold from histogram — pure numba, no cv2."""
    hist = np.zeros(256, dtype=np.int32)
    H, W = gray.shape
    for r in range(H):
        for c in range(W):
            hist[gray[r, c]] += 1
    total = H * W
    sum_all = 0.0
    for i in range(256):
        sum_all += i * hist[i]
    sumB = 0.0; wB = 0; best = 0.0; thresh = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0: continue
        wF = total - wB
        if wF == 0: break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        var = wB * wF * (mB - mF) * (mB - mF)
        if var > best:
            best = var; thresh = t
    return thresh


@njit(parallel=True, cache=True)
def threshold_combined(gray, t_otsu, t_mean):
    """
    Single parallel pass: pixel=255 if gray < t_otsu OR gray < t_mean.
    Replaces two separate cv2.threshold calls + bitwise_or.
    """
    H, W = gray.shape
    out = np.empty((H, W), dtype=np.uint8)
    t_max = max(t_otsu, t_mean)   # OR of two inversions = single threshold at max
    for r in prange(H):
        for c in range(W):
            out[r, c] = 255 if gray[r, c] <= t_max else 0
    return out


@njit(parallel=True, cache=True)
def otsu_and_threshold(gray):
    """
    Single function: compute Otsu+mean threshold AND apply binarization.
    One serial histogram pass + one parallel threshold pass.
    Avoids the overhead of cv2.threshold just for the Otsu value.
    Returns binary uint8 array (0/1, not 0/255).
    """
    H, W = gray.shape
    # --- histogram (serial, 256 bins) ---
    hist = np.zeros(256, dtype=np.int32)
    for r in range(H):
        for c in range(W):
            hist[gray[r, c]] += 1
    total = H * W
    # --- otsu ---
    sum_all = 0.0
    for i in range(256):
        sum_all += i * hist[i]
    sumB = 0.0; wB = 0; best = 0.0; t_otsu = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0: continue
        wF = total - wB
        if wF == 0: break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        var = wB * wF * (mB - mF) * (mB - mF)
        if var > best:
            best = var; t_otsu = t
    # --- mean ---
    t_mean = int(sum_all / total)
    t_max = max(t_otsu, t_mean)
    # --- parallel threshold (0/1 output for open_lines_numba) ---
    out = np.empty((H, W), dtype=np.uint8)
    for r in prange(H):
        for c in range(W):
            out[r, c] = np.uint8(1) if gray[r, c] <= t_max else np.uint8(0)
    return out


@njit(cache=True)
def open_lines_serial(b, L):
    """Single-threaded numba — consistent latency, no thread spin-up jitter."""
    H, W = b.shape
    out_h = np.zeros((H, W), dtype=np.uint8)
    out_v = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        row = b[r]
        c = 0
        while c <= W - L:
            if row[c] == 1:
                run = 1
                while run < L and row[c + run] == 1:
                    run += 1
                if run == L:
                    start, end = c, c + run
                    while end < W and row[end] == 1:
                        end += 1
                    for k in range(start, end):
                        out_h[r, k] = 1
                    c = end
                    continue
            c += 1
    for col in range(W):
        r = 0
        while r <= H - L:
            if b[r, col] == 1:
                run = 1
                while run < L and b[r + run, col] == 1:
                    run += 1
                if run == L:
                    start, end = r, r + run
                    while end < H and b[end, col] == 1:
                        end += 1
                    for k in range(start, end):
                        out_v[k, col] = 1
                    r = end
                    continue
            r += 1
    return out_h | out_v


@njit(parallel=True, cache=True)
def open_lines_numba(b, L):
    """
    Combined horizontal + vertical morphological OPEN via numba parallel JIT.
    Single function, both directions, no intermediate array allocation.
    Each row (h-open) and each column (v-open) processed in parallel.
    """
    H, W = b.shape
    out_h = np.zeros((H, W), dtype=np.uint8)
    out_v = np.zeros((H, W), dtype=np.uint8)

    # horizontal open (parallel over rows)
    for r in prange(H):
        row = b[r]
        c = 0
        while c <= W - L:
            # find next run of L ones
            if row[c] == 1:
                run = 1
                while run < L and row[c + run] == 1:
                    run += 1
                if run == L:
                    # mark all pixels in this run (and extend for dilation)
                    start = c
                    end = c + run
                    # extend run as long as 1s continue (capture full run for dilation)
                    while end < W and row[end] == 1:
                        end += 1
                    for k in range(start, end):
                        out_h[r, k] = 1
                    c = end
                    continue
            c += 1

    # vertical open (parallel over cols)
    for col in prange(W):
        r = 0
        while r <= H - L:
            if b[r, col] == 1:
                run = 1
                while run < L and b[r + run, col] == 1:
                    run += 1
                if run == L:
                    start = r
                    end = r + run
                    while end < H and b[end, col] == 1:
                        end += 1
                    for k in range(start, end):
                        out_v[k, col] = 1
                    r = end
                    continue
            r += 1

    return out_h | out_v
