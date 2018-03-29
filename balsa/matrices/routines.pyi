from typing import Union
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