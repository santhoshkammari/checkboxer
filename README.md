# checkboxer

> Pure NumPy checkbox detection. No deep learning. No OpenCV dependency. Just fast.

## What is this?

**checkboxer** detects checkboxes in images and returns their bounding boxes — built purely on NumPy.

Most existing solutions rely on OpenCV, YOLOv8, or heavy ML frameworks. This project proves you don't need any of that.

```python
from checkboxer import detect

bboxes = detect("form.png")
# Returns: [(x, y, w, h), ...]
```

## Why?

| Library | Dependency | Speed |
|---|---|---|
| boxdetect | OpenCV | slow |
| YOLOv8-based | PyTorch + CUDA | heavy |
| checkbox_detection (OpenCV) | OpenCV | moderate |
| **checkboxer** | **NumPy only** | **fast** |

Existing solutions felt either too heavy or not optimized for raw speed. This is my attempt to build the fastest checkbox detector possible using first principles — just NumPy and image math.

## Goal

- Input: image (PNG/JPG)
- Output: list of bounding boxes `[(x, y, w, h), ...]`
- Zero OpenCV, zero PyTorch, zero deep learning
- Benchmarked against popular alternatives

## Benchmarks (coming soon)

Will compare speed, accuracy, and memory usage against:
- [boxdetect](https://github.com/karolzak/boxdetect) (OpenCV-based)
- [LynnHaDo/Checkbox-Detection](https://github.com/LynnHaDo/Checkbox-Detection) (YOLOv8)
- [edbabayan/Checkbox-Detection](https://github.com/edbabayan/Checkbox-Detection) (ML-based)
- [Manikandan/checkbox_detection_opencv](https://github.com/Manikandan-Thangaraj-ZS0321/checkbox_detection_opencv) (OpenCV)

## Install

```bash
pip install checkboxer
```

Or from source:

```bash
git clone https://github.com/santhoshkammari/checkboxer
cd checkboxer
pip install -e .
```

## Status

🔨 Active development — building the core detector now.

## Author

[santhoshkammari](https://github.com/santhoshkammari)
