import pandas as pd
from typing import Union, Iterable, List

def fast_stack(frame: pd.DataFrame, multi_index: pd.MultiIndex, deep_copy: bool=True) -> pd.Series:
    pass

def fast_unstack(series: pd.Series, index: pd.Index, columns: pd.Index, deep_copy: bool=True) -> pd.DataFrame:
    pass

def align_categories(iterable: Iterable[Union[pd.Series, pd.DataFrame]]) -> None:
    pass

def split_zone_in_matrix(base_matrix: pd.DataFrame, old_zone: int, new_zones: List[int], proportions: List[float]
                         ) -> pd.DataFrame:
    pass

def sum_df_sequence(seq: Iterable[pd.DataFrame], fill_value: Union[int, float]=0) -> pd.DataFrame:
    pass
