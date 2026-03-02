import numpy as np
from numba import njit, prange, set_num_threads
import os

# pin to physical cores to reduce thread scheduling jitter
_CORES = max(1, os.cpu_count() // 2)
set_num_threads(_CORES)


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
