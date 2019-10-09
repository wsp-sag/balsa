import numpy as np
import pandas as pd
from contextlib import contextmanager

from six import string_types

try:
    from pathlib import Path
except ImportError:
    Path = None


def coerce_matrix(matrix, allow_raw=True, force_square=True):
    """
    Infers a NumPy array from given input

    Args:
        matrix:
        allow_raw (bool, optional): Defaults to ``True``.
        force_square (bool, optional): Defaults to ``True``.

    Returns:
        numpy.ndarray:
            A 2D ndarray of type float32
    """
    if isinstance(matrix, pd.DataFrame):
        if force_square:
            assert matrix.index.equals(matrix.columns)
        return matrix.values.astype(np.float32)
    elif isinstance(matrix, pd.Series):
        assert matrix.index.nlevels == 2, "Cannot infer a matrix from a Series with more or fewer than 2 levels"
        wide = matrix.unstack()

        union = wide.index | wide.columns
        wide = wide.reindex_axis(union, fill_value=0.0, axis=0).reindex_axis(union, fill_value=0.0, axis=1)
        return wide.values.astype(np.float32)

    if not allow_raw:
        raise NotImplementedError()

    matrix = np.array(matrix, dtype=np.float32)
    assert len(matrix.shape) == 2
    i, j = matrix.shape
    assert i == j

    return matrix


def expand_array(a, n, axis=None):
    """
    Expands an array across all dimensions by a set amount

    Args:
        a (numpy.ndarray): The array to expand
        n (numpy.ndarray): The (non-negative) number of items to expand by.
        axis (int, optional): Defaults to ``None``. The axis to expand along, or None to expand along all axes.

    Returns:
        numpy.ndarray:
            The expanded array
    """

    if axis is None: new_shape = [dim + n for dim in a.shape]
    else:
        new_shape = []
        for i, dim in enumerate(a.shape):
            dim += n if i == axis else 0
            new_shape.append(dim)

    out = np.zeros(new_shape, dtype=a.dtype)

    indexer = [slice(0, dim) for dim in a.shape]
    out[indexer] = a

    return out


@contextmanager
def open_file(file_handle, **kwargs):
    """
    Context manager for opening files provided as several different types. Supports a file handler as a str, unicode,
    ``pathlib.Path``, or an already-opened handler.

    Args:
        file_handle (Union[str, unicode, Path, File]): The item to be opened or is already open.
        **kwargs: Keyword args passed to ``open()``. Usually mode='w'.

    Yields:
        File:
            The opened file handler. Automatically closed once out of context.

    """
    opened = False
    if isinstance(file_handle, string_types):
        f = open(file_handle, **kwargs)
        opened = True
    elif Path is not None and isinstance(file_handle, Path):
        f = file_handle.open(**kwargs)
        opened = True
    else:
        f = file_handle

    try:
        yield f
    finally:
        if opened:
            f.close()
