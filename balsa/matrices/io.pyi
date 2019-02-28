from typing import Union, List, Any, Iterable, Dict, Tuple
from io import FileIO

import numpy as np
import pandas as pd

try:
    from pathlib import Path
    file_type = Union[str, FileIO, Path]
except ImportError:
    Path = None
    file_type = Union[str, FileIO]

def coerce_matrix(matrix, allow_raw=True, force_square=True): pass

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


def to_fortran(matrix: Union[pd.Series, pd.DataFrame, np.ndarray], file: file_type, n_columns: int=None,
               min_index: int=1, forec_square: bool=True): pass

MATRIX_TYPES = Union[pd.DataFrame, pd.Series, np.ndarray]

def read_omx(file: Union[Path, str], matrices: Iterable[str]=None, mapping: str=None, raw=False, tall=False,
             squeeze=True) -> Union[MATRIX_TYPES, Dict[str, MATRIX_TYPES]]: pass

def _check_types(matrices: Dict[str, MATRIX_TYPES]) -> str: pass

def _check_raw_matrices(matrices: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int]: pass

def _check_matrix_series(matrices: Dict[str, pd.Series]) -> Tuple[Dict[str, np.ndarray], pd.Index]: pass

def _check_matrix_frames(matrices: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, np.ndarray], pd.Index]: pass

def _prep_matrix_dict(matrices: Dict[str, MATRIX_TYPES], desired_zone_index: pd.Index
                      ) -> Tuple[Dict[str, np.ndarray], pd.Index]: pass

def to_omx(file: str, matrices: Dict[str, MATRIX_TYPES], zone_index: pd.Index=None, title: str='',
               descriptions: Dict[str, str]=None,  attrs: Dict[str, dict]=None, mapping: str='zone_numbers'): pass
