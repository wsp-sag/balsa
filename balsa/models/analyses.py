import pandas as pd
import numpy as np
import numexpr as ne


def tlfd(values, bin_start=0, bin_end=200, bin_step=2, weights=None, intrazonal=None, label_type='MULTI',
                 include_top=False):
    """
    Generates a Trip Length Frequency Distribution (i.e. a histogram) from given data. Produces a "pretty" Pandas object
    suitable for charting.

    Args:
        values (ndarray or Series): Vector of trip lengths, with a length  of "N". Can be provided from a table of
            trips, or from a matrix (in "tall" format).
        bin_start (int): The minimum bin value, in the same units as `values`. Default is 0.
        bin_end (int): The maximum bin value, in the same units as `values`. Defaults to 200. Values over this limit
            are either ignored, or counted under a separate category (see `include top`)
        bin_step (int): The size of each bin, in the same unit as `values`. Default is 2.
        weights (ndarray, Series, or None: Optional vector of weights to use of length "N", to produce a weighted
            histogram.
        intrazonal (ndarray, Seires, or None): Optional boolean vector indicating which values are considered
            "intrazonal". When specified, prepends an "intrazonal" category to the front of the histogram.
        label_type (str): String indicating the format of the returned index. Options are:
            - "MULTI": The returned index will be a 2-level MultiIndex ['from', 'to'];
            - "TEXT": The returned index will be text-based: "0 to 2";
            - "BOTTOM": The returned index will be the bottom of each bin; and
            - "TOP": The returned index will be the top of each bin.
        include_top (bool): If True, the function will count all values (and weights, if provided) above the `bin_top`,
            and add them to the returned Series. This bin is described as going from `bin_top` to `inf`.

    Returns:
        Series: The weighted or unweighted histogram, depending on the options configured above.

    """
    bins = list(range(bin_start, bin_end + bin_step, bin_step))

    if intrazonal is not None:
        if weights is not None:
            iz_total = weights.loc[intrazonal].sum()
            weights = weights.loc[~intrazonal]
        else:
            iz_total = intrazonal.sum()

        values = values.loc[~intrazonal]

    if weights is not None:
        hist, _ = np.histogram(values, weights=weights, bins=bins)
    else:
        hist, _ = np.histogram(values, bins=bins)

    new_len = len(hist)
    if intrazonal is not None: new_len += 1
    if include_top: new_len += 1
    new_hist = np.zeros(shape=new_len, dtype=hist.dtype)
    lower_index = 0
    upper_index = new_len

    if intrazonal is not None:
        new_hist[0] = iz_total
        bins.insert(0, 'intrazonal')
        lower_index += 1
    if include_top:
        if weights is not None:
            top = weights.loc[values >= bin_end].sum()
        else:
            top = (values >= bin_end).sum()

        new_hist[-1] = top
        bins.append(np.inf)
        upper_index -= 1
    new_hist[lower_index: upper_index] = hist

    label_type = label_type.upper()
    if label_type == 'MULTI':
        index = pd.MultiIndex.from_arrays([bins[:-1], bins[1:]], names=['from', 'to'])
    elif label_type == 'TOP':
        index = pd.Index(bins[1:])
    elif label_type == 'BOTTOM':
        index = pd.Index(bins[:-1])
    elif label_type == 'TEXT':
        s0 = pd.Series(bins[:-1], dtype=str).astype(str)
        s1 = pd.Series(bins[1:], dtype=str).astype(str)
        index = pd.Index(s0 + ' to ' + s1)
    else:
        raise NotImplementedError(label_type)

    new_hist = pd.Series(new_hist, index=index)

    return new_hist


def distance_matrix(x, y, tall=False, method='euclidean', coord_unit=1.0, labels=None):
    """
    Fastest method of computing a distance matrix from 2 vectors of coordinates, using the NumExpr package. Can compute
    Manhattan distance as well as straight-line.

    Args:
        x (ndarray or Series): Vector of x-coordinates, of length N. Can be a Series to specify labels.
        y (ndarray or Series): Vector of y-coordinates, of length N. Can be a Series to specify labels.
        tall (bool): If True, returns a vecotr whose shape is (N x N,). Otherwise, returns a matrix whose shape is
            (N, N).
        method (str): Specifies the method by which to compute distance. Valid options are:
            'EUCLIDEAN': Computes straight-line, 'as-the-crow flies' distance.
            'MANHATTAN': Computes the Manhattan distance
        coord_unit (float): Factor applies directly to the result, defaulting to 1.0 (no conversion). Useful when the
            coordinates are provided in one unit (e.g. m) and the desired result is in a different unit (e.g. km).
        labels (None or list or Index): Optional labels for each item in the x, y vectors; of length N.

    Returns:
        Series: Returned when `tall=True`, and labels can be inferred (see note below). Will always be have 2-level
            MultiIndex.
        DataFrame: Returned when `tall=False` and labels can be inferred (see notes below).
        ndarray: Returned when labels could not be inferred (see notes below). If `tall=True` the array will be
            1-dimensional, with shape (N x N,). Otherwise, it will 2-dimensional with shape (N, N)

    Notes:
        The type of the returned object depends on whether labels can be inferred from the arguments. This is always
        true when the `labels` argument is specified, and the returned value will use cross-product of the `labels`
        vector.

        Otherwise, the function will try and infer the labels from the `x` and `y` objects, if one or both of them are
        provided as Series.

    """

    x_is_series, y_is_series = isinstance(x, pd.Series), isinstance(y, pd.Series)

    if x_is_series and y_is_series:
        assert x.index.equals(y.index), "X and Y series must have the same index"
        if labels is None: labels = x.index
        x0, y0 = x.values[...], y.values[...]
    elif x_is_series:
        assert len(y) == len(x), "The length of the Y array does not match the length of the X series"
        if labels is None: labels = x.index
        x0, y0 = x.values[...], y[...]
    elif y_is_series:
        assert len(y) == len(x), "The length of the X array does not match the length of the Y series"
        if labels is None: labels = y.index
        x0, y0 = x[...], y.values[...]
    else:
        assert len(x) == len(y), "X and Y arrays are not the same length"
        if labels is not None: assert len(labels) == len(x), "Vector length of labels does not match X/Y vector length"
        x0, y0 = x[...], y[...]

    x1, y1 = x0[...], y0[...]
    n = len(x0)

    x0.shape = 1, n
    y0.shape = 1, n
    x1.shape = n, 1
    y1.shape = n, 1

    if method.lower() == 'euclidean':
        expr = "sqrt((x0 - x1)**2 + (y0 - y1) ** 2) * coord_unit"
    elif method.lower() == 'manhattan':
        expr = "(abs(x0 - x1) + abs(y0 - y1)) * coord_unit"
    else:
        # TODO: Support Haversine approach in which coords are in Lat/Lon
        raise NotImplementedError(method.lower())

    raw_matrix = ne.evaluate(expr)

    if tall:
        raw_matrix.shape = n * n
        if labels is None: return raw_matrix

        mi = pd.MultiIndex.from_product([labels, labels])
        return pd.Series(raw_matrix, index=mi)
    elif labels is None:
        return raw_matrix

    return pd.DataFrame(raw_matrix, index=labels, columns=labels)

