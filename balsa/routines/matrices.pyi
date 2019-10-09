from typing import Union, Callable, Tuple, Iterable, List
import numpy as np
import pandas as pd


def matrix_balancing_1d(m: np.ndarray, a: np.ndarray, axis: int) -> np.ndarray: pass


def matrix_balancing_2d(m: np.ndarray, a: np.ndarray, b: np.ndarray, max_iterations: int=1000, rel_error: float=0.0001,
                        n_procs: int=1) -> np.ndarray: pass


def matrix_bucket_rounding(m: Union[np.ndarray, pd.DataFrame], decimals: int=0) -> Union[np.ndarray, pd.DataFrame]:
    pass


def split_zone_in_matrix(base_matrix: pd.DataFrame, old_zone: int, new_zones: List[int], proportions: List[float]
                         ) -> pd.DataFrame:
    pass

Num = Union[int, float]
Vector = Union[pd.Series, np.ndarray]

def aggregate_matrix(matrix: Union[pd.DataFrame, pd.Series],
                     groups: Vector=None,  row_groups: Vector=None, col_groups: Vector=None,
                     aggfunc: Callable[[Iterable[Num]], Num]=np.sum) -> Union[pd.DataFrame, pd.Series]:
    pass


def fast_stack(frame: pd.DataFrame, multi_index: pd.MultiIndex, deep_copy: bool=True) -> pd.Series:
    pass


def fast_unstack(series: pd.Series, index: pd.Index, columns: pd.Index, deep_copy: bool=True) -> pd.DataFrame:
    pass

def disaggregate_matrix(matrix: pd.DataFrame, mapping: pd.Series = None, proportions: pd.Series = None,
                        row_mapping: pd.Series = None, row_proportions: pd.Series = None, col_mapping: pd.Series = None,
                        col_proportions: pd.Series = None) -> pd.DataFrame:
    pass