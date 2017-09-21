from typing import Iterable, List, Tuple, Any, Set, Union, Dict
import pandas as pd
import numpy as np

class _ChoiceNode:

    _root: 'ChoiceTree'
    _parent: Union['ChoiceTree', '_ChoiceNode']
    _name: str
    _level: int
    _children: Set['_ChoiceNode']

    def __init__(self, *args):
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def parent(self) -> bool:
        pass

    @property
    def level(self) -> int:
        pass

    @property
    def is_parent(self) -> bool:
        pass

    def max_level(self) -> int:
        pass

    def children(self) -> Iterable['_ChoiceNode']:
        pass

    def add_node(self, name: str, logsum_scale: float=1.0) -> '_ChoiceNode':
        pass

class ChoiceTree:

    _children: Set[_ChoiceNode]
    _all_nodes: Dict[str, _ChoiceNode]

    def __init__(self, *args):
        pass

    def max_level(self) -> int:
        pass

    def children(self) -> Iterable['_ChoiceNode']:
        pass

    @property
    def node_index(self) -> pd.Index:
        pass

    def add_node(self, name: str, logsum_scale: float=1.0) -> _ChoiceNode:
        pass

    def add_nodes(self, names: List[str], logsum_scales: List[float]=None) -> List[_ChoiceNode]:
        pass

    def remove_node(self, name: str):
        pass

    def flatten(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def _root_add(self, parent, new_name, logsum_scale, level) -> _ChoiceNode:
        pass
