from typing import Union, Iterable, Dict

import pandas as pd
import numpy as np

from .common import file_type

MATRIX_TYPES = Union[pd.DataFrame, pd.Series, np.ndarray]

def read_omx(file: file_type, matrices: Iterable[str]=None, mapping: str=None, raw=False, tall=False,
             squeeze=True) -> Union[MATRIX_TYPES, Dict[str, MATRIX_TYPES]]: pass

def to_omx(file: str, matrices: Dict[str, MATRIX_TYPES], zone_index: pd.Index=None, title: str='',
               descriptions: Dict[str, str]=None,  attrs: Dict[str, dict]=None, mapping: str='zone_numbers'): pass