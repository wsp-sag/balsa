"""
LinkedDataFrame core
"""
import pandas as pd
import numpy as np

from collections import namedtuple
from six import iteritems, itervalues
from balsa.utils import name_is_pythonic

LinkageEntry = namedtuple("LinkageEntry", ['other_frame', 'self_indexer', 'other_indexer', 'fill_value',
                                           'self_names', 'self_index_flag', 'other_names', 'other_index_flag',
                                           'aggregation_required'])

SUPPORTED_AGGREGATIONS = {
    'count', 'first', 'last', 'max', 'min', 'mean', 'median', 'prod', 'std', 'sum', 'var'
}


class LinkageSpecificationError(ValueError):
    pass


class LinkageNode(object):
    """
    Linkage derived from LinkedDataFrame
    """

    def __init__(self, frame, root_index, history):
        self._history = history
        self._df = frame
        self._root_index = root_index

    def __dir__(self):
        return [col for col in self._df.columns if name_is_pythonic(col)] + list(self._df._links.keys())

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __getattr__(self, item):
        if item in self._df._links:
            return self._get_link(item)
        if item not in self._df:
            raise AttributeError(item)

        s = self._df[item]
        for left_indexer, right_indexer, fill_value in self._history:
            s.index = right_indexer
            s = s.reindex(left_indexer, fill_value=fill_value)
        s.index = self._root_index
        return s

    def _get_link(self, item):

        linkage_entry = self._df._links[item]

        history_copy = list(self._history) # Make a copy
        history_copy.insert(0, (linkage_entry.self_indexer, linkage_entry.other_indexer, linkage_entry.fill_value))

        other = linkage_entry.other_frame
        if linkage_entry.aggregation_required:
            return LinkageAttributeAggregator(other, self._root_index, history_copy)
        if isinstance(other, LinkedDataFrame):
            return LinkageNode(other, self._root_index, history_copy)
        return LinkageLeaf(other, self._root_index, history_copy)


class LinkageLeaf(object):
    """
    Linkage derived from regular DataFrame.
    """

    def __init__(self, frame, root_index, history):
        self._history = history
        self._df = frame
        self._root_index = root_index

    def __dir__(self):
        return self._df.columns

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __getattr__(self, item):
        if item not in self._df.columns:
            raise AttributeError(item)

        s = self._df[item]
        for left_indexer, right_indexer, fill_value in self._history:
            s.index = right_indexer
            s = s.reindex(left_indexer, fill_value=fill_value)
        s.index = self._root_index
        return s


class LinkageAttributeAggregator(object):
    """
    Many-to-one linkage manager.
    """

    def __init__(self, frame, root_index, history):
        self._df = frame
        self._history = history
        self._root_index = root_index

    def __dir__(self):
        return dir(LinkageAttributeAggregator) + list(self._df.columns)

    def _apply(self, func, expr):
        self_indexer, other_indexer, fill_value = self._history[0]

        s = self._df.eval(expr)
        if not isinstance(s, pd.Series):
            # Sometimes a single value is returned, so ensure that it is cast to Series
            s = pd.Series(s, index=self._df.index)
        s.index = other_indexer
        grouper = 0 if other_indexer.nlevels == 1 else other_indexer.names
        grouped = s. groupby(level=grouper)

        if func == 'first':
            s = grouped.first()
        elif func == 'last':
            s = grouped.last()
        else:
            s = grouped.aggregate(func)
        s = s.reindex(self_indexer, fill_value=fill_value)

        for left_indexer, right_indexer, fill_value in self._history[1:]:
            s.index = right_indexer
            s = s.reindex(left_indexer, fill_value=fill_value)
        s.index = self._root_index
        return s

    def count(self, expr="1"):
        """
        Analagous to pandas.GroupBy.count()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series

        """
        return self._apply(np.count_nonzero, expr)

    def first(self, expr="1"):
        """
        Analagous to pandas.GroupBy.first()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series

        """
        return self._apply('first', expr)

    def last(self, expr="1"):
        """
        Analagous to pandas.GroupBy.last()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series

        """
        return self._apply('last', expr)

    def max(self, expr="1"):
        """
        Analagous to pandas.GroupBy.max()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series
        """
        return self._apply(np.max, expr)

    def mean(self, expr="1"):
        """
        Analagous to pandas.GroupBy.mean()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series
        """
        return self._apply(np.mean, expr)

    def median(self, expr="1"):
        """
        Analagous to pandas.GroupBy.median()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series
        """
        return self._apply(np.median, expr)

    def min(self, expr="1"):
        """
        Analagous to pandas.GroupBy.min()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series
        """
        return self._apply(np.min, expr)

    def prod(self, expr="1"):
        """
        Analagous to pandas.GroupBy.prod()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series
        """
        return self._apply(np.prod, expr)

    def std(self, expr="1"):
        """
        Analagous to pandas.GroupBy.std()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series
        """
        return self._apply(np.std, expr)

    def sum(self, expr="1"):
        """
        Analagous to pandas.GroupBy.sum()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series
        """
        return self._apply(np.sum, expr)

    def var(self, expr="1"):
        """
        Analagous to pandas.GroupBy.var()

        Args:
            expr (basestring): An attribute or expression to evaluate in the context of this DataFrame

        Returns: Series
        """
        return self._apply(np.var, expr)


def _is_aggregation_required(self_indexer, other_indexer):
    """
    Optimized code to test if Linkage relationship is many-to-one or one-to-one.

    Args:
        self_indexer:
        other_indexer:

    Returns: bool

    """
    # If the right indexer is 100% unique, then no aggregation is required
    if other_indexer.is_unique:
        return False

    # Otherwise, aggregation is only required if at least one duplicate value
    # is in the left indexer.
    dupes = other_indexer.get_duplicates()
    for dupe in dupes:
        # Eager loop through the index. For this to be slow, a large number of duplicates must be missing
        # in self_indexer, which is practically never the case.
        if dupe in self_indexer:
            return True
    return False


def _get_indexer_from_frame(frame, names, get_from_index):
    """
    Gets or constructs an indexer object from a given DataFrame

    Args:
        frame (DataFrame):
        names (list): A list of names. If None, this function returns the index of the frame.
        get_from_index (bool): True if new indexer is to be generated from the index of the frame,
            False if it is to be generated from the columns of the DataFrame

    Returns: Either an Index or MultiIndex, depending on the arguments

    Raises: LinkageSpecificationError

    """
    if names is None: return frame.index

    if get_from_index:
        if len(names) > frame.index.nlevels:
            raise LinkageSpecificationError("Cannot specify more levels than in the index")
        arrays = []
        for name in names:
            # `get_level_values` works on both Index and MultiIndex objects and accepts both
            # integer levels AND level names
            try: arrays.append(frame.index.get_level_values(name))
            except KeyError: raise LinkageSpecificationError("'%s' not in index" % name)
    else:
        arrays = []
        for name in names:
            if name not in frame: raise LinkageSpecificationError("'%s' not in columns" % name)
            arrays.append(frame[name])

    if len(arrays) == 1:
        return pd.Index(arrays[0], name=names[0])
    return pd.MultiIndex.from_arrays(arrays, names=names)


class LinkedDataFrame(pd.DataFrame):
    """
    Subclass of DataFrame, which links to other DataFrames. This allows "chained" attribute access.

    To create a link between a LinkedDataFrame and another DataFrame, call the `link(other, alias, *args, **kwargs)`
    method. Links can be established either on level(s) in the index, or on column(s) in either Frame. The other Frame
    will be set to the given alias.
    """

    @property
    def _constructor(self):
        return LinkedDataFrame

    def __init__(self, *args, **kwargs):
        super(LinkedDataFrame, self).__init__(*args, **kwargs)
        object.__setattr__(self, '_links', {})
        object.__setattr__(self, '_pythonic_links', set())

    def __finalize__(self, original_ldf, method=None, **kwargs):
        """ Essential functionality to ensure that slices of LinkedDataFrame retain the same behaviour """
        pd.DataFrame.__finalize__(self, original_ldf, method=method, **kwargs)

        self._links = {}
        for alias, linkage_entry in iteritems(original_ldf._links):
            self_names = linkage_entry.self_names
            get_self_indexer_from_index = linkage_entry.self_index_flag

            new_self_indexer = _get_indexer_from_frame(self, self_names, get_self_indexer_from_index)

            new_linkage_entry = LinkageEntry(linkage_entry.other_frame, new_self_indexer, linkage_entry.other_indexer,
                                             linkage_entry.fill_value,
                                             linkage_entry.self_names, linkage_entry.self_index_flag,
                                             linkage_entry.other_names, linkage_entry.other_index_flag,
                                             linkage_entry.aggregation_required)

            self._links[alias] = new_linkage_entry
        self._pythonic_links = set(original_ldf._pythonic_links)

        return self

    def __dir__(self):
        """ Override dir() to show links as valid attributes """
        return super(LinkedDataFrame, self).__dir__() + sorted(self._pythonic_links)

    def __getitem__(self, item):
        if item in self._links:
            return self._get_link(item)
        return super(LinkedDataFrame, self).__getitem__(item)

    def __getattr__(self, item):
        if item in self._links:
            return self._get_link(item)
        return super(LinkedDataFrame, self).__getattr__(item)

    def link_to(self, other, alias, on=None, on_self=None, on_other=None, levels=None, self_levels=None, other_levels=None,
                fill_value=np.NaN):
        """
        Creates a new link from this DataFrame to another, assigning it to the given name.

        The relationship between the left-hand-side (this DataFrame itself) and the right-hand-side (the other
        DataFrame) must be pre-specified to create the link. The relationship can be based on the index (or a subset of
        it levels in a MultiIndex) OR based on columns in either DataFrame.

        By default, if both the "levels" and "on" args of one side are None, then the join will be made on ALL levels
        of the side's index.

        Regardless of whether the join is based on an index or columns, the same number of levels must be given. For
        example, if the left-hand indexer uses two levels from the index, then the right-hand indexer must also use
        two levels in the index or two columns.s

        When the link is established, it is tested whether the relationship is one-to-one or many-to-one (the latter
        indicates that aggregation is required). The result of this test is returned by this method.

        Args:
            other (DataFrame or LinkedDataFrame):
            alias (basestring):  The alias (symbolic name) of the new link. If Pythonic, this will show up as an
                attribute; otherwise the link will need to be accessed using [].
            on (list or basestring or None): If given, the join will be made on the provided **column(s)** in both this
                and the other DataFrame. This arg cannot be used with `levels` and will override `on_self` and
                `on_other`.
            on_self (list or basestring or None): If provided, the left-hand side of the join will be made on the
                column(s) in this DataFrame. This arg cannot be used with `self_levels`.
            on_other: (list or basestring or None): If provided, th right-hand-side of the join will be made on the
                column(s) in the other DataFrame. This arg cannot be used with `other_levels`.
            levels (list or basestring or None): If provided, the join will be made on the given **level(s)**
                in both this and the other DataFrame's index. It can be specified as an integer or a string,
                if both indexes have the same level names. This arg cannot be used with `on` and will override
                `self_levels` and `other_levels`.
            self_levels (list or basestring or None): If provided, the left-hand-side of the join will be made on the
                level(s) in this DataFrame. This arg cannot be used with `on_self`.
            other_levels (list or basestring or None): If provided, the right-hand-side of the join will be made on the
                level(s) in the other DataFrame. This arg cannot be used with `on_other`.
            fill_value (numeric): The value to fill in the results if there are any missing values. Defaults to NaN.

        Returns:
            bool: True if aggregation is required for this link. False otherwise.

        Raises:
            LinkageSpecificationError

        """

        if isinstance(on, str): on = [on]
        if isinstance(on_self, str): on_self = [on_self]
        if isinstance(on_other, str): on_other = [on_other]
        if isinstance(levels, str): levels = [levels]
        if isinstance(self_levels, str): self_levels = [self_levels]
        if isinstance(other_levels, str): other_levels = [other_levels]

        self_indexer_names = None
        get_self_indexer_from_index = True
        other_indexer_names = None
        get_other_indexer_from_index = True

        if on is not None and levels is not None:
            raise LinkageSpecificationError()
        elif on is not None:
            self_indexer_names = on
            get_self_indexer_from_index = False
            other_indexer_names = on
            get_other_indexer_from_index = False
        elif levels is not None:
            self_indexer_names = levels
            get_self_indexer_from_index = True
            other_indexer_names = levels
            get_other_indexer_from_index = True
        else:
            if on_self is not None and self_levels is not None:
                raise LinkageSpecificationError()
            elif on_self is not None:
                self_indexer_names = on_self
                get_self_indexer_from_index = False
            elif self_levels is not None:
                self_indexer_names = self_levels
                get_self_indexer_from_index = True

            if on_other is not None and other_levels is not None:
                raise LinkageSpecificationError()
            elif on_other is not None:
                other_indexer_names = on_other
                get_other_indexer_from_index = False
            elif other_levels is not None:
                other_indexer_names = other_levels
                get_other_indexer_from_index = True

        self_indexer = _get_indexer_from_frame(self, self_indexer_names, get_self_indexer_from_index)
        other_indexer = _get_indexer_from_frame(other, other_indexer_names, get_other_indexer_from_index)

        aggregation_flag = _is_aggregation_required(self_indexer, other_indexer)

        link = LinkageEntry(other, self_indexer, other_indexer, fill_value,
                            self_indexer_names, get_self_indexer_from_index,
                            other_indexer_names, get_other_indexer_from_index,
                            aggregation_flag)

        self._links[alias] = link

        if name_is_pythonic(alias):
            self._pythonic_links.add(alias)

        return aggregation_flag

    def _get_link(self, name):

        linkage_entry = self._links[name]

        history = [(linkage_entry.self_indexer, linkage_entry.other_indexer, linkage_entry.fill_value)]

        other = linkage_entry.other_frame
        if linkage_entry.aggregation_required:
            return LinkageAttributeAggregator(other, self.index, history)
        if isinstance(other, LinkedDataFrame):
            return LinkageNode(other, self.index, history)
        return LinkageLeaf(other, self.index, history)

    def remove_link(self, alias):
        """
        Destroys a linkage.

        Args:
            alias (basestring): The name of the linkage to remove.

        Raises: KeyError if no linkage matches the given alias.

        """
        del self._links[alias]
        if alias in self._pythonic_links:
            self._pythonic_links.remove(alias)

    def refresh_links(self):
        """
        Updates all outgoing linkages, as there is no callback when the other indexer changes. In practise, this doesn't
        happen very often - the most frequent example is if `set_index(*args, inplace=True)` is called. The join might
        break if it is based on the index.

        Raises:
            LinkageSpecificationError
        """
        for linkage_entry in itervalues(self._links):
            linkage_entry.other_indexer = _get_indexer_from_frame(linkage_entry.other_frame,
                                                                  linkage_entry.other_names,
                                                                  linkage_entry.other_index_flag)
