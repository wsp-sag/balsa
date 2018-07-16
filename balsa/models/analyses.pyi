from typing import Iterable, Union
from pandas import DataFrame, Series, Index
from numpy import ndarray


_vector_type = Union[ndarray, Series]


def tlfd(values: _vector_type, bin_start: int=0, bin_end: int=200, bin_step: int=2, weights: _vector_type=None,
         intrazonal: _vector_type=None, label_type: str='MULTI', include_top: bool=False
         ) -> Series:
    pass


def distance_matrix(x: _vector_type, y: _vector_type, tall: bool=False, method: str='euclidean', coord_unit: float=1.0,
                    labels: Union[Iterable, Index]=None
                    ) -> Union[ndarray, Series, DataFrame]:
    pass
