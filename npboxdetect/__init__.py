name = "npboxdetect"
__version__ = "0.1.0"

# pre-compile numba kernel on import with a tiny dummy array
import numpy as _np
from npboxdetect._numba_ops import open_lines_numba as _w
_w(_np.zeros((4, 4), dtype=_np.uint8), 2)
