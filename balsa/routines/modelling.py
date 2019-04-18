import pandas as pd
import numpy as np
import numexpr as ne
from six import iteritems


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


def _get_distance_equation(method):
    if method.lower() == 'euclidean':
        expr = "sqrt((x0 - x1)**2 + (y0 - y1) ** 2) * coord_unit"
    elif method.lower() == 'manhattan':
        expr = "(abs(x0 - x1) + abs(y0 - y1)) * coord_unit"
    elif method.lower() == 'haversine':
        y0 = "(y0 * pi / 180)"
        y1 = "(y1 * pi / 180)"
        delta_lon = "((x1 - x0) * pi / 180)"
        delta_lat = "((y1 - y0) * pi / 180)"
        part1 = "sin({delta_lat} / 2.0)**2 + cos({y0}) * cos({y1}) * (sin({delta_lon} / 2.0))**2"\
            .format(delta_lat=delta_lat, delta_lon=delta_lon, y0=y0, y1=y1)
        expr = "6371.0 * earth_radius_factor* 2.0 * arctan2(sqrt({part1}), sqrt(1.0 - {part1})) * coord_unit".\
            format(part1=part1)
    else:
        raise NotImplementedError(method.lower())
    return expr


def _prepare_distance_kwargs(kwargs):
    defaults = {'coord_unit': 1.0, 'earth_radius_factor': 1.0, 'pi': np.pi}
    for key, val in iteritems(defaults):
        if key not in kwargs:
            kwargs[key] = val


def _check_vectors(description: str, *vectors):
    if len(vectors) < 1: return []

    first = vectors[0]
    retval = []
    common_index, common_length = None, None
    if isinstance(first, pd.Series):
        common_index = first.index
        common_length = len(common_index)
        retval.append(first.values[...])
    else:
        retval.append(first[...])

    for vector in vectors[1:]:
        if isinstance(vector, pd.Series):
            assert vector.index.equals(common_index), "All %s Series must have the same index" % description
            retval.append(vector.values[...])
        else:
            assert len(vector) == common_length, "All %s vectors must have the same length" % description
            retval.append(vector[...])

    return common_index, retval


def distance_matrix(x0, y0, tall=False, method='euclidean', labels0=None, x1=None, y1=None, labels1=None, **kwargs):
    """
    Fastest method of computing a distance matrix from vectors of coordinates, using the NumExpr package. Supports
    several equations for computing distances.

    Accepts two or four vectors of x-y coordinates. If only two vectors are provided (x0, y0), the result will be the
    2D product of this vector with itself (vector0 * vector0). If all four are provided (x0, y0, x1, y1), the result
    will be the 2D product of the first and second vector (vector0 * vector1).

    Args:
        x0 (ndarray or Series): Vector of x-coordinates, of length N0. Can be a Series to specify labels.
        y0 (ndarray or Series): Vector of y-coordinates, of length N0. Can be a Series to specify labels.
        tall (bool): If True, returns a vector whose shape is N0 x N1. Otherwise, returns a matrix whose shape is
            (N0, N1).
        method (str): Specifies the method by which to compute distance. Valid options are:
            'EUCLIDEAN': Computes straight-line, 'as-the-crow flies' distance.
            'MANHATTAN': Computes the Manhattan distance
            'HAVERSINE': Computes distance based on lon/lat.
        labels0 (Index-like): Override set of labels to use if x0 and y0 are both raw Numpy arrays
        x1 (ndarray or Series): Optional second vector of x-coordinates, of length N1. Can be a Series to specify labels
        y1 (ndarray or Series): Optional second vector of y-coordinates, of length N1. Can be a Series to specify labels
        labels1 (Index-like): Override set of labels to use if x1 and y1 are both raw Numpy arrays

        **kwargs: Additional scalars to pass into the evaluation context
            coord_unit (float): Factor applies directly to the result, defaulting to 1.0 (no conversion). Useful when
                the coordinates are provided in one unit (e.g. m) and the desired result is in a different unit (e.g.
                km). Only used for Euclidean or Manhattan distance
            earth_radius_factor (float): Factor to convert from km to other units when using Haversine distance

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

    second_coords = x1 is not None and y1 is not None

    descr = "first coordinate" if second_coords else "coordinate"
    temp_labels, (x_array0, y_array0) = _check_vectors(descr, x0, y0)
    if labels0 is None: labels0 = temp_labels

    if second_coords:
        temp_labels, (x_array1, y_array1) = _check_vectors("second coordinate", x1, y1)
        if labels1 is None: labels1 = temp_labels
    else:
        x_array1 = x_array0[...]
        y_array1 = y_array0[...]
        labels1 = labels0

    n0, n1 = len(x_array0), len(x_array1)

    x_array0.shape = n0, 1
    y_array0.shape = n0, 1
    x_array1.shape = 1, n1
    y_array1.shape = 1, n1

    expr = _get_distance_equation(method)
    kwargs = kwargs.copy()
    _prepare_distance_kwargs(kwargs)
    kwargs['x0'] = x_array0
    kwargs['x1'] = x_array1
    kwargs['y0'] = y_array0
    kwargs['y1'] = y_array1

    raw_matrix = ne.evaluate(expr, local_dict=kwargs)
    labelled_result = labels0 is not None and labels1 is not None

    if tall:
        raw_matrix.shape = n0 * n1
        if not labelled_result: return raw_matrix

        mi = pd.MultiIndex.from_product([labels0, labels1])
        return pd.Series(raw_matrix, index=mi)
    elif not labelled_result:
        return raw_matrix

    return pd.DataFrame(raw_matrix, index=labels0, columns=labels1)


def distance_array(x0, y0, x1, y1, method='euclidean', **kwargs):
    """
    Fast method to compute distance between 2 (x, y) points, represented by 4 separate arrays, using the NumExpr
    package. Supports several equations for computing distances

    Args:
        x0: X or Lon coordinate of first point
        y0: Y or Lat coordinate of first point
        x1: X or Lon coordinate of second point
        y1: Y or Lat coordinate of second point
        method: method (str): Specifies the method by which to compute distance. Valid options are:
            'EUCLIDEAN': Computes straight-line, 'as-the-crow flies' distance.
            'MANHATTAN': Computes the Manhattan distance
            'HAVERSINE': Computes distance based on lon/lat.
        **kwargs: Additional scalars to pass into the evaluation context
            coord_unit (float): Factor applies directly to the result, defaulting to 1.0 (no conversion). Useful when
                the coordinates are provided in one unit (e.g. m) and the desired result is in a different unit (e.g.
                km). Only used for Euclidean or Manhattan distance
            earth_radius_factor (float): Factor to convert from km to other units when using Haversine distance

    Returns:
        ndarray: Distance from the vectors of first points to the vectors of second points.
        Series: Distance from the vectors of first points to the vectors of second points, when one or more coordinate
            arrays are given as a Series object

    """

    labels, (x0, y0, x1, y1) = _check_vectors("coordinate", x0, y0, x1, y1)

    expr = _get_distance_equation(method)
    kwargs = kwargs.copy()
    _prepare_distance_kwargs(kwargs)
    kwargs['x0'] = x0
    kwargs['x1'] = x1
    kwargs['y0'] = y0
    kwargs['y1'] = y1

    result_array = ne.evaluate(expr, local_dict=kwargs)

    if labels is not None:
        return pd.Series(result_array, index=labels)

    return result_array
