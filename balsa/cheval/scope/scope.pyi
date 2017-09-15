import pandas as pd
import numpy as np
from typing import Iterable, Union, Any
from balsa.cheval.ldf import LinkedDataFrame

class Scope:

    _root: Any
    _empty_symbols: set
    _filled_symbols: set
    _records: pd.Index
    _alternatives: pd.Index

    def __init__(self, model):
        pass

    def set_record_index(self, index: Union[pd.Index, Iterable]):
        pass

    def fill_symbol(self, symbol_name: str,
                    data: Union[int, float, np.ndarray, pd.Series, pd.DataFrame, LinkedDataFrame],
                    orientation: int=None, strict: bool=True, allow_unused: bool=True):
        pass

    def _initialize(self):
        pass

    def _compute_utilities(self, n_threads, logger=None):
        pass

    def _evaluate_single_expression(self, expr, utility_table):
        pass

    def _fill_simple(self, data, orientation=None, strict=True):
        pass

    @staticmethod
    def _check_orientation(orientation):
        pass

    def _check_records(self):
        pass

    def _fill_attributed(self, data, orientation=None):
        pass

    def _fill_linked(self, data, orientation):
        pass

    def clear(self):
        pass

    def _fill_dict_literals(self, dict_literals):
        pass

    def _symbolize(self):
        pass
