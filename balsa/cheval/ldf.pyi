from typing import Any, List, Dict, Set, Iterator

from pandas import DataFrame, Index
import numpy as np


class LinkageEntry(object):
    def __init__(self, *args, **kwargs):
        self.other_frame: DataFrame
        self.self_indexer: Index
        self.other_indexer: Index
        self.fill_value: Any
        self.self_names: List[str]
        self.self_index_flag: bool
        self.other_names: List[str]
        self.other_index_flag: bool
        self.aggregation_required: bool


class LinkedDataFrame(DataFrame):

    def __init__(self, *args, **kwargs):
        self._links: Dict[str, LinkageEntry]
        self._pythonic_links: Set[str]

    def link_to(self, other: DataFrame, alias: str, on: str=None, on_self: str=None, on_other: str=None,
                levels: str=None, self_levels: str=None, other_levels: str=None, fill_value: Any=np.NaN) -> bool: pass

    def remove_link(self, alis: str): pass

    def links(self) -> Iterator[str]: pass
