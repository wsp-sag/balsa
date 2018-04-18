from __future__ import division, absolute_import, print_function, unicode_literals

from pandas import DataFrame, Series, Index, MultiIndex
import numpy as np
from six import iteritems


def fast_stack(frame, multi_index, deep_copy=True):
    """
    Performs the same action as DataFrame.stack(), but provides better performance when the target stacked index is
    known before hand. Useful in converting a lot of matrices from "wide" to "tall" format. The inverse of fast_unstack()

    Notes:
        This function does not check that the entries in the multi_index are compatible with the index and columns of the
        source DataFrame, only that the lengths are compatible. It can therefore be used to assign a whole new set of
        labels to the result.

    Args:
        frame (DataFrame): DataFrame to stack.
        multi_index (Index): The 2-level MultiIndex known ahead-of-time.
        deep_copy (bool): Flag indicating if the returned Series should be a view of the underlying data
            (deep_copy=False) or a copy of it (deep_copy=True). A deep copy takes a little longer to convert and takes
            up more memory but preserves the original data of the DataFrame. The default value of True is recommended
            for most uses.

    Returns:
        Series: The stacked data.

    """

    assert multi_index.nlevels == 2, "Target index must be a MultiIndex with exactly 2 levels"
    assert len(multi_index) == len(frame.index) * len(frame.columns), "Target index and source index and columns do " \
                                                                      "not have compatible lengths"

    array = np.ascontiguousarray(frame.values)
    array = array.copy() if deep_copy else array[:, :]
    array.shape = len(frame.index) * len(frame.columns)

    return Series(array, index=multi_index)


def fast_unstack(series, index, columns, deep_copy=True):
    """
    Performs the same action as DataFrame.unstack(), but provides better performance when the target unstacked index and
    columns are known before hand. Useful in converting a lot of matrices from "tall" to "wide" format. The inverse of
    fast_stack().

    Notes:
        This function does not check that the entries in index and columns are compatible with the MultiIndex of the
        source Series, only that the lengths are compatible. It can therefore be used to assign a whole new set of
        labels to the result.

    Args:
        series (Series): Series with 2-level MultiIndex to stack()
        index (Index): The row index known ahead-of-time
        columns (Index): The columns index known ahead-of-time.
        deep_copy (bool): Flag indicating if the returned DataFrame should be a view of the underlying data
            (deep_copy=False) or a copy of it (deep_copy=True). A deep copy takes a little longer to convert and takes
            up more memory but preserves the original data of the Series. The default value of True is recommended
            for most uses.

    Returns:
        DataFrame: The unstacked data

    """

    assert series.index.nlevels == 2, "Source Series must have an index with exactly 2 levels"
    assert len(series) == len(index) * len(columns), "Source index and target index and columns do not have " \
                                                     "compatible lengths"

    array = series.values.copy() if deep_copy else series.values[:]
    array.shape = len(index), len(columns)

    return DataFrame(array, index=index, columns=columns)


def reindex_series(series, target_series, source_levels=None, target_levels=None, fill_value=None):

    # Make shallow copies of the source and target series in case their indexes need to be changed
    series = series.copy(deep=False)
    target_series = target_series.copy(deep=False)

    if series.index.nlevels > 1 and source_levels is not None:
        arrays = [series.index.get_level_values(level) for level in source_levels]
        series.index = MultiIndex.from_arrays(arrays)

    if target_series.index.nlevels > 1 and target_levels is not None:
        arrays = [target_series.index.get_level_values(level) for level in target_levels]
        target_series.index = MultiIndex.from_arrays(arrays)

    reindexed = series.reindex(target_series.values, fill_value=fill_value)
    reindexed.index = target_series.index

    return reindexed


def align_categories(iterable):
    """
    Pre-processing step for pandas.concat() which attempts to align any Categorical series in the sequence to using the
    same set of categories. It passes through the sequence twice: once to accumulate the complete set of all categories
    used in the sequence; and a second time to modify the sequence's contents to use this full set. The contents of the
    sequence are modified in-place.

    Notes:
        The resulting categories will be lex-sorted (based on the sorted() builtin)

    Args:
        iterable: Any iterable of Series or DataFrame objects (anything that is acceptable to pandas.concat())

    """
    iterable_type = None
    for item in iterable:
        if iterable_type is None:
            if isinstance(item, DataFrame): iterable_type = DataFrame
            elif isinstance(item, Series): iterable_type = Series
            else: raise TypeError(type(item))
        else:
            assert isinstance(item, iterable_type)

    if iterable_type is Series:
        _align_series_categories(iterable)
    else:
        column_categories = _enumerate_frame_categories(iterable)
        _align_frame_categories(iterable, column_categories)

    return


def _align_series_categories(series_list):
    all_categories = set()
    for series in series_list:
        if not hasattr(series, 'cat'):
            raise TypeError()
        all_categories |= set(series.cat.categories)

    sorted_categories = sorted(all_categories)
    for series in series_list:
        missing_categories = all_categories.difference(series.cat.categories)
        if missing_categories:
            series.cat.add_categories(missing_categories, inplace=True)
        series.cat.reorder_categories(sorted_categories, inplace=True)


def _enumerate_frame_categories(frames):
    column_categories = {}
    for frame in frames:
        for col_name, series in frame.items():
            if not hasattr(series, 'cat'): continue
            categories = set(series.cat.categories)

            if col_name not in column_categories:
                column_categories[col_name] = categories
            else:
                column_categories[col_name] |= categories
    return column_categories


def _align_frame_categories(frames, column_categories):
    for col_name, all_categories in iteritems(column_categories):
        sorted_categories = sorted(all_categories)
        for frame in frames:
            if col_name not in frame: continue
            s = frame[col_name]
            missing_categories = all_categories.difference(s.cat.categories)
            if missing_categories:
                s.cat.add_categories(missing_categories, inplace=True)
            s.cat.reorder_categories(sorted_categories, inplace=True)


def sum_df_sequence(seq, fill_value=0):
    """
    Sums over a sequence of DataFrames, even if they have different indexes or columns, filling in 0 (or a value of your
    choice) for missing rows or columns. Useful when you have a sequence of DataFrames which are supposed to have
    the same indexes and columns but might be missing a few values.

    Args:
        seq (Iterable[DataFrame]): Any iterable of DataFrame type, ordered or unordered.
        fill_value: The value fo use for missing cells. Preferably a number to avoid erros.

    Returns:
        DataFrame: The sum over all items in seq.

    """
    common_index = Index([])
    common_columns = Index([])
    accumulator = DataFrame()

    for df in seq:
        if not df.index.equals(common_index):
            common_index |= df.index
            accumulator = accumulator.reindex_axis(common_index, axis=0, fill_value=fill_value)
            df = df.reindex_axis(common_index, axis=0, fill_value=fill_value)
        if not df.columns.equals(common_columns):
            common_columns |= df.columns
            accumulator = accumulator.reindex_axis(common_columns, axis=1, fill_value=fill_value)
            df = df.reindex_axis(common_columns, axis=1, fill_value=fill_value)
        accumulator += df
    return accumulator
