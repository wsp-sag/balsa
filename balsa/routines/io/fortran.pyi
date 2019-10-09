from typing import Any, Union

import pandas as pd
import numpy as np

from .common import file_type


def read_fortran_rectangle(file: file_type, n_columns: int, zones: pd.Index=None, tall: bool=False,
                           reindex_rows: bool=False, fill_value: Any=None
                           ) -> Union[pd.Series, pd.DataFrame, np.ndarray]: pass

def read_fortran_square(file: file_type, zones: pd.Index=None, tall: bool=False
                        ) -> Union[pd.Series, pd.DataFrame, np.ndarray]: pass


def to_fortran(matrix: Union[pd.Series, pd.DataFrame, np.ndarray], file: file_type, n_columns: int=None,
               min_index: int=1, forec_square: bool=True): pass