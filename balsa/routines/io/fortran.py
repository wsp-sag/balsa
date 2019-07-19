"""
FORTRAN optimized format
========================

For working with .bin files commonly used by certain legacy FORTRAN applications.

"""

import pandas as pd
import numpy as np

from .common import coerce_matrix, open_file, expand_array


def _infer_fortran_zones(n_words):
    """Returns the inverse of n_words = matrix_size * (matrix_size + 1)"""
    n = int(0.5 + ((1 + 4 * n_words)**0.5)/2) - 1
    assert n_words == (n * (n + 1)), "Could not infer a square matrix from file"
    return n


def read_fortran_rectangle(file, n_columns, zones=None, tall=False, reindex_rows=False, fill_value=None):
    """
    Reads a FORTRAN-friendly .bin file (a.k.a. 'simple binary format') which is known to NOT be square. Also works with
    square matrices.

    This file format is an array of 4-bytes, where each row is prefaced by an integer referring to the 1-based positional
    index that FORTRAN uses. The rest of the data are in 4-byte floats. To read this, the number of columns present
    must be known, since the format does not self-specify.

    Args:
        file(str or File or Path): The file to read.
        n_columns (int): The number of columns in the matrix.
        zones (None or int or pandas.Index): An Index or Iterable will be interpreted as the zone labels for the matrix
            rows and columns; returning a DataFrame or Series (depending on `tall`). If an integer is provided, the
            returned ndarray will be truncated to this 'number of zones'.
        tall (bool): If true, a 'tall' version of the matrix will be returned.
        reindex_rows (bool): If true, and zones is an Index, the returned DataFrame will be reindexed to fill-in any
            missing rows.
        fill_value: The value to pass to pandas.reindex()

    Returns:
        ndarray or DataFrame or Series

    Raises:
        AssertionError if the shape is not valid.
    """
    with open_file(file, mode='rb') as reader:
        n_columns = int(n_columns)

        matrix = np.fromfile(reader, dtype=np.float32)
        rows = len(matrix) // (n_columns + 1)
        assert len(matrix) == (rows * (n_columns + 1))

        matrix.shape = rows, n_columns + 1

        # Convert binary representation from float to int, then subtract 1 since FORTRAN uses 1-based positional
        # indexing
        row_index = np.frombuffer(matrix[:, 0].tobytes(), dtype=np.int32) - 1
        matrix = matrix[:, 1:]

        if zones is None:
            if tall:
                matrix.shape = matrix.shape[0] * matrix.shape[1]
            return matrix

        if isinstance(zones, (int, np.int_)):
            matrix = matrix[: zones, :zones]

            if tall:
                matrix.shape = zones * zones
            return matrix

        nzones = len(zones)
        matrix = matrix[: nzones, : nzones]
        row_labels = zones.take(row_index[:nzones])
        matrix = pd.DataFrame(matrix, index=row_labels, columns=zones)

        if reindex_rows:
            matrix = matrix.reindex_axis(zones, axis=0, fill_value=fill_value)

        if tall:
            return matrix.stack()
        return matrix


def read_fortran_square(file, zones=None, tall=False):
    """
    Reads a FORTRAN-friendly .bin file (a.k.a. 'simple binary format') which is known to be square.

    This file format is an array of 4-bytes, where each row is prefaced by an integer referring to the 1-based positional
    index that FORTRAN uses. The rest of the data are in 4-byte floats. To read this, the number of columns present
    must be known, since the format does not self-specify. This method can infer the shape if it is square.

    Args:
        file (str or File or Path): The file to read.
        zones (Index or int or None): An Index or Iterable will be interpreted as the zone labels for the matrix rows
            and columns; returning a DataFrame or Series (depending on `tall`). If an integer is provided, the returned
            ndarray will be truncated to this 'number of zones'. Otherwise, the returned ndarray will be size to the
            maximum number of zone dimensioned by the Emmebank.
        tall (bool):  If True, a 1D data structure will be returned. If `zone_index` is provided, a Series will be
            returned, otherwise a 1D ndarray.

    Returns:
        DataFrame or ndarray

    """
    with open_file(file, mode='rb') as reader:
        floats = np.fromfile(reader, dtype=np.float32)
        n_words = len(floats)
        matrix_size = _infer_fortran_zones(n_words)
        floats.shape = matrix_size, matrix_size + 1

        data = floats[:, 1:]

        if zones is None:
            if tall:
                n = np.prod(data.shape)
                data.shape = n
                return data
            return data

        if isinstance(zones, (int, np.int_)):
            data = data[:zones, :zones]

            if tall:
                data.shape = zones * zones
                return data
            return data
        elif zones is None:
            return data

        zones = pd.Index(zones)
        n = len(zones)
        data = data[:n, :n]

        matrix = pd.DataFrame(data, index=zones, columns=zones)

        return matrix.stack() if tall else matrix


def to_fortran(matrix, file, n_columns=None, min_index=1, force_square=True):
    """
    Reads a FORTRAN-friendly .bin file (a.k.a. 'simple binary format'), in a square format.

    Args:
        matrix (DataFrame or Series or ndarray): The matrix to write to disk. If a Series is given, it MUST have a
            MultiIndex with exactly 2 levels to unstack.
        file (basestring or File): The path or file handler to write to.
        n_columns (int): Optionally specify a desired "width" of the matrix file. For example, n_columns=4000 on a
            matrix 3500x3500 will pad the width with 500 extra columns containing 0. If None if provided or if the
            number of columns <= the width of the given matrix, no padding will be performed.
        min_index (int): The lowest numbered row. Used when slicing matrices
        force_square (bool):

    """
    assert min_index >= 1
    array = coerce_matrix(matrix, force_square=force_square)

    if n_columns is not None and n_columns > array.shape[1]:
        extra_columns = n_columns - array.shape[1]
        array = expand_array(array, extra_columns, axis=1)

    with open_file(file, mode='wb') as writer:
        rows, columns = array.shape
        temp = np.zeros([rows, columns + 1], dtype=np.float32)
        temp[:, 1:] = array

        index = np.arange(min_index, rows + 1, dtype=np.int32)
        # Mask the integer binary representation as floating point
        index_as_float = np.frombuffer(index.tobytes(), dtype=np.float32)
        temp[:, 0] = index_as_float

        temp.tofile(writer)
