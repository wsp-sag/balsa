from typing import Union, List, Any
from io import FileIO

import numpy as np
import pandas as pd

try:
    from pathlib import Path
    file_type = Union[str, FileIO, Path]
except ImportError:
    Path = None
    file_type = Union[str, FileIO]

def _coerce_matrix(matrix, allow_raw=True): pass

def expand_array(a: np.ndarray, n: int, axis: int=None) -> np.ndarray: pass

def read_mdf(file: file_type, raw: bool=False, tall: bool=False) -> Union[pd.DataFrame, np.ndarray]: pass

def to_mdf(matrix: Union[pd.Series, pd.DataFrame], file: file_type): pass

def peek_mdf(file: file_type, as_index: bool=True) -> Union[List[pd.Index], List[List[int]]]: pass

def read_emx(file: file_type, zones: pd.Index=None, tall: bool=False) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    pass

def to_emx(matrix: Union[pd.Series, pd.DataFrame, np.ndarray], file: file_type, emmebank_zones: int): pass

def _infer_fortran_zones(n_words): pass

def read_fortran_rectangle(file: file_type, n_columns: int, zones: pd.Index=None, tall: bool=False,
                           reindex_rows: bool=False, fill_value: Any=None
                           ) -> Union[pd.Series, pd.DataFrame, np.ndarray]: pass

def read_fortran_square(file: file_type, zones: pd.Index=None, tall: bool=False
                        ) -> Union[pd.Series, pd.DataFrame, np.ndarray]: pass


def to_fortran(matrix: Union[pd.Series, pd.DataFrame, np.ndarray], file: file_type, n_columns: int=None): pass
