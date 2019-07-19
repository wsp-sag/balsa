from typing import Union, List, Any, Iterable, Dict, Tuple

import pandas as pd
import numpy as np

from .common import file_type

def read_mdf(file: file_type, raw: bool=False, tall: bool=False) -> Union[pd.DataFrame, np.ndarray]: pass

def to_mdf(matrix: Union[pd.Series, pd.DataFrame], file: file_type): pass

def peek_mdf(file: file_type, as_index: bool=True) -> Union[List[pd.Index], List[List[int]]]: pass

def read_emx(file: file_type, zones: pd.Index=None, tall: bool=False) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    pass

def to_emx(matrix: Union[pd.Series, pd.DataFrame, np.ndarray], file: file_type, emmebank_zones: int): pass
