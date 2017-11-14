import pandas as pd
import numpy as np


def tlfd_from_records(records, length_col, weight_col=None, bin_min=0.0, bin_max=100.0, bin_step=2.0,
                      normalized=True, ceiling=True, intrazonal_col=None, adjacent_col=None, bins=None,
                      pretty_labels=False):

    if length_col not in records: raise KeyError("Length column '%s' not in the trip records" % length_col)
    length_col = records[length_col]

    if weight_col is not None: weight_col = records[weight_col]
    else: weight_col = pd.Series(1.0, index=records.index)

    if intrazonal_col is not None:
        intrazonal_col = records[intrazonal_col].astype(bool)
        intrazonal_total = weight_col.loc[intrazonal_col].sum()
        records = records.loc[~intrazonal_col]

    if adjacent_col is not None:
        adjacent_col = records[adjacent_col].astype(bool)
        adjacent_total = weight_col.loc[adjacent_col].sum()
        records = records.loc[~adjacent_col]

    if bins is None:
        bins = np.arange(bin_min, bin_max, bin_step)

    histogram_array, output_bins = np.histogram(length_col, bins, weights=weight_col)
    if pretty_labels:
        bin_labels = ["%s to %s" % low_high for low_high in zip(output_bins[:-1], output_bins[1:])]
    else:
        bin_labels = [low_high for low_high in zip(output_bins[:-1], output_bins[1:])]
    histogram_series = pd.Series(histogram_array, index=bin_labels)

    to_concat = []
    if intrazonal_col is not None:
        to_concat.append(pd.Series({'intrazonal': intrazonal_total}))
    if adjacent_col is not None:
        to_concat.append(pd.Series({'adjacent': adjacent_total}))
    to_concat.append(histogram_series)
    if ceiling:
        max_length = output_bins[-1]
        above_max = length_col >= max_length
        total_max = weight_col.loc[above_max].sum()
        label = "More than %s" % max_length if pretty_labels else (max_length, 'inf')
        to_concat.append(pd.Series({label: total_max}))

    tlfd = pd.concat(to_concat, axis=0)
    if normalized:
        total = tlfd.sum()
        return tlfd / total
    return tlfd
