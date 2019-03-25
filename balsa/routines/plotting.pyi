from typing import List, Union, Dict, Callable, Tuple, Any, Optional

from pandas import DataFrame, Series
from matplotlib.axes import Axes

import six
if six.PY3:
    from pathlib import Path as PathType
else:
    PathType = str


def convergence_boxplot(
        targets: DataFrame,
        results: DataFrame,
        filter_func: Callable[[Series], Series],
        adjust_target: bool=True,
        percentage: bool=True,
        band: Tuple[float, float]=None,
        simple_labels: bool=True,
        ax=None,
        fp: str=None,
        title: str=None
        ) -> Axes:
    pass


def location_summary(
        model: DataFrame,
        target: DataFrame,
        ensemble_names: Series,
        title: str='',
        fp: PathType=None,
        dpi: int=150,
        district_name: str='Ensemble'
        ) -> Axes:
    pass


def trumpet_diagram(
        counts: Series,
        model_volume: Series,
        categories: Union[Series, List[Series]]=None,
        category_colours: Dict[Union[Any, tuple]]=None,
        category_markers: Dict[Union[Any, tuple]]=None,
        label_format: str=None,
        title: str='',
        y_bounds: Tuple[float, float]=(-2, 2),
        ax: Optional[Axes]=None,
        alpha: float=1.0,
        x_label: str="Count volume"
        ) -> Axes:
    pass
