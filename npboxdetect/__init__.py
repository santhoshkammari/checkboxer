name = "npboxdetect"
__version__ = "0.1.1"

# pre-compile numba kernel on import with a tiny dummy array
import numpy as _np
from npboxdetect._numba_ops import open_lines_numba as _w, threshold_combined as _tc, otsu_threshold as _ot, otsu_and_threshold as _oat
_w(_np.zeros((4, 4), dtype=_np.uint8), 2)
_tc(_np.zeros((4, 4), dtype=_np.uint8), 100, 120)
_ot(_np.zeros((4, 4), dtype=_np.uint8))
_oat(_np.zeros((4, 4), dtype=_np.uint8))

from npboxdetect.detector import get_boxes
