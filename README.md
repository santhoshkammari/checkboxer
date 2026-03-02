# checkboxer / npboxdetect

> Fastest checkbox detector. NumPy + Numba. No deep learning.

## Benchmark

| Approach | p50 (ms) | boxes | vs npboxdetect |
|---|---|---|---|
| **npboxdetect** | **34ms** | 69 | baseline |
| boxdetect | 60ms | 67 | 1.8x slower |
| opencv_contours | 33ms | 179 | ~same (but noisy) |
| morphology | 34ms | 25 | ~same (misses many) |

*50-run p50, full pipeline including imread, on 2200×1700 document images.*

**Pure processing (imread excluded):**

| Image | npboxdetect | boxdetect | Speedup |
|---|---|---|---|
| lc_application1.png | **6.5ms** | 25.5ms | **3.9x** |
| lc_application2.png | **7.1ms** | 18.6ms | **2.6x** |

## Usage

```python
from npboxdetect.detector import get_boxes

boxes = get_boxes("form.png")
# [(x, y, w, h), ...]
```

## How it works

Five steps, all near-hardware-limit:

```
downsample 2x        ~0ms   stride view, zero copy
otsu + threshold      1.4ms  fused numba: histogram → binarize in one shot
morph open            1.3ms  numba parallel run-length scan (rows+cols)
connected components  1.3ms  cv2.connectedComponentsWithStats
NMS                   0.2ms  vectorized pairwise IoU matrix
─────────────────────────────
processing total      ~6ms
```

Key ideas:
- **2x downsample** before processing — free stride view, halves pixel count
- **Fused numba otsu+threshold** — one serial histogram pass + one parallel binarize pass, no cv2 call
- **Run-length morph open** — each row/column independent → `prange` parallelism, no cumsum overhead
- **Filter before NMS** — size/ratio filter drops 1000+ blobs to ~70 before NMS

## Install

```bash
git clone https://github.com/santhoshkammari/checkboxer
cd checkboxer
pip install numba opencv-python numpy
```

## Journey

See [JOURNEY.md](JOURNEY.md) (v0→v6, 4248ms→37ms) and [JOURNEY2.md](JOURNEY2.md) (v7→v10, 37ms→6ms).

## Author

[santhoshkammari](https://github.com/santhoshkammari)
