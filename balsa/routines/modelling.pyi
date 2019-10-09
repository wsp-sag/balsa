from typing import Iterable, Union, Tuple, Optional
from pandas import DataFrame, Series, Index
from numpy import ndarray


_vector_type = Union[ndarray, Series]


def tlfd(values: _vector_type, bin_start: int=0, bin_end: int=200, bin_step: int=2, weights: _vector_type=None,
         intrazonal: _vector_type=None, label_type: str='MULTI', include_top: bool=False
         ) -> Series:
    pass


def distance_matrix(x: _vector_type, y: _vector_type, tall: bool=False, method: str='euclidean',
                    labels0: Union[Iterable, Index]=None, x1: _vector_type=None, y1: _vector_type=None,
                    labels1: Union[Iterable, Index]=None, earth_radius_factor: float=1.0, coord_unit: float=1.0
                    ) -> Union[ndarray, Series, DataFrame]:
    pass


def distance_array(x0: _vector_type, y0: _vector_type, x1: _vector_type, y1: _vector_type, method: str='euclidean',
                   earth_radius_factor: float=1.0, coord_unit: float=1.0):
    pass

def indexers_for_map_matrix(row_labels: Series, col_labels: Series, superset: Index, check=True
                            ) -> Tuple[ndarray, ndarray]:
    pass


def map_to_matrix(values: Series, super_labels: Index, fill_value=0, *,
                  row_col_labels: Optional[Tuple[Series, Series]] = None,
                  row_col_offsets: Optional[Tuple[ndarray, ndarray]] = None,
                  out: Optional[Union[DataFrame, ndarray]] = None,
                  grouper_func='sum',  out_operand: str='+'
                  ) -> DataFrame:
    pass
