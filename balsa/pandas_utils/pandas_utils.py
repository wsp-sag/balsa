from __future__ import division, absolute_import, print_function, unicode_literals

from pandas import DataFrame, Series, Index, MultiIndex
import numpy as np


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
    all_categories = set()
    for series_or_frame in iterable:
        if isinstance(series_or_frame, Series):
            all_categories |= set(series_or_frame.cat.categories)
        elif isinstance(series_or_frame, DataFrame):
            for column in series_or_frame:
                s = series_or_frame[column]
                try:
                    all_categories |= set(s.cat.categories)
                except AttributeError:
                    # Skip series with non-categorical dtype
                    pass
        else:
            # Do nothing, although concat() might not allow this
            pass
    sorted_categories = sorted(all_categories)

    for series_or_frame in iterable:
        if isinstance(series_or_frame, Series):
            missing_categories = all_categories - set(series_or_frame.cat.categories)
            if missing_categories:
                series_or_frame.cat.add_categories(missing_categories)
            series_or_frame.cat.reorder_categories(sorted_categories, inplace=True)
        elif isinstance(series_or_frame, DataFrame):
            for column in series_or_frame:
                s = series_or_frame[column]
                try:
                    missing_categories = all_categories - set(s.cat.categories)
                except AttributeError:
                    pass
                else:
                    if missing_categories:
                        s.cat.add_categories(missing_categories)
                    s.car.reorder_categories(sorted_categories, inplace=True)

    return


def split_zone_in_matrix(base_matrix, old_zone, new_zones, proportions):
    """
    Takes a zone in a matrix (represented as a DataFrame) and splits it into several new zones,
    prorating affected cells by a vector of proportions (one value for each new zone). The old
    zone is removed.

    Args:
        base_matrix: The matrix to re-shape, as a DataFrame
        old_zone: Integer number of the original zone to split
        new_zones: List of integers of the new zones to add
        proportions: List of floats of proportions to split the original zone to. Must be the same
            length as `new_zones` and sum to 1.0

    Returns: Re-shaped DataFrame
    """

    assert isinstance(base_matrix, DataFrame), "Base matrix must be a DataFrame"

    old_zone = int(old_zone)
    new_zones = np.array(new_zones, dtype=np.int32)
    proportions = np.array(proportions, dtype=np.float64)

    assert len(new_zones) == len(proportions), "Proportion array must be the same length as the new zone array"
    assert len(new_zones.shape) == 1, "New zones must be a vector"
    assert base_matrix.index.equals(base_matrix.columns), "DataFrame is not a matrix"
    assert np.isclose(proportions.sum(), 1.0), "Proportions must sum to 1.0 "

    n_new_zones = len(new_zones)

    intersection_index = base_matrix.index.drop(old_zone)
    new_index = intersection_index
    for z in new_zones: new_index = new_index.insert(-1, z)
    new_index = Index(sorted(new_index))

    new_matrix = DataFrame(0, index=new_index, columns=new_index, dtype=base_matrix.dtypes.iat[0])

    # 1. Copy over the values from the regions of the matrix not being updated
    new_matrix.loc[intersection_index, intersection_index] = base_matrix

    # 2. Prorate the row corresponding to the dropped zone
    # This section (and the next) works with the underlying Numpy arrays, since they handle
    # broadcasting better than Pandas does
    original_row = base_matrix.loc[old_zone, intersection_index]
    original_row = original_row.values[:] # Make a shallow copy to preserve shape of the original data
    original_row.shape = 1, len(intersection_index)
    proportions.shape = n_new_zones, 1
    result = DataFrame(original_row * proportions, index=new_zones, columns=intersection_index)
    new_matrix.loc[result.index, result.columns] = result

    # 3. Proprate the column corresponding to the dropped zone
    original_column = base_matrix.loc[intersection_index, old_zone]
    original_column = original_column.values[:]
    original_column.shape = len(intersection_index), 1
    proportions.shape = 1, n_new_zones
    result = DataFrame(original_column * proportions, index=intersection_index, columns=new_zones)
    new_matrix.loc[result.index, result.columns] = result

    # 4. Expand the old intrazonal
    proportions_copy = proportions[:,:]
    proportions_copy.shape = 1, n_new_zones
    proportions.shape = n_new_zones, 1

    intrzonal_matrix = proportions * proportions_copy
    intrazonal_scalar = base_matrix.at[old_zone, old_zone]

    result = DataFrame(intrazonal_scalar * intrzonal_matrix, index=new_zones, columns=new_zones)
    new_matrix.loc[result.index, result.columns] = result

    return new_matrix

