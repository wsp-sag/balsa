from typing import Any
import pandas as pd


def tlfd_from_records(records: pd.DataFrame, length_col: str, weight_col: str=None, bin_min: float=0.0,
                      bin_max: float=100.0, bin_step: float=2.0,  normalized: bool=True, ceiling: bool=True,
                      intrazonal_col: str=None, adjacent_col: str=None, bins: Any=None, pretty_labels: bool=False
                      ) -> pd.DataFrame: pass
