from typing import Union
from io import FileIO

import numpy as np

try:
    from pathlib import Path
    file_type = Union[str, FileIO, Path]
except ImportError:
    Path = None
    file_type = Union[str, FileIO]

def coerce_matrix(matrix, allow_raw=True, force_square=True): pass

def expand_array(a: np.ndarray, n: int, axis: int=None) -> np.ndarray: pass

def open_file(file_handle, **kwargs): pass
