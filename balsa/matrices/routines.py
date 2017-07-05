"""
Core Matrix Manipulation Tools
==============================

==========================================================================
Matrix Balancing
==========================================================================
matrix_balancing_1d    Singly-constrained matrix balancing
matrix_balancing_2d    Doubly-constrained balancing using
                       iterative-proportional fitting
==========================================================================

"""

from __future__ import division as _division

import multiprocessing as _mp
import numpy as _np
import pandas as _pd


def matrix_balancing_1d(m, a, axis):
    """ Balances a matrix using a single constraint.

    Args:
        m (numpy ndarray (M, M)): Matrix to be balanced
        a (numpy ndarray (M)): Totals
        axis (int): Direction to constrain (0 = along columns, 1 = along rows)

    Return:
        w :  Numpy ndarray(..., M, M)
    """

    assert axis in [0, 1], "axis must be either 0 or 1"
    assert m.ndim == 2, "m must be a two-dimensional matrix"
    assert a.ndim == 1, "a must be a two-dimensional matrix"
    assert m.shape[axis] == a.shape[0], "axis %d of matrice 'm' and 'a' must be the same." % axis

    return _balance(m, a, axis)


def matrix_balancing_2d(m, a, b, totals_to_use='raise', max_iterations=1000,
                        rel_error=0.0001, n_procs=1):
    """ Balances a two-dimensional matrix using iterative proportional fitting.

    Args:
        m (numpy ndarray (M, M): Matrix to be balanced
        a (numpy ndarray (M)): Row totals
        b (numpy ndarray (M)): Column totals
        totals_to_use (str, optional):
            Describes how to scale the row and column totals if their sums do not match
            Must be one of ['rows', 'columns', 'average', 'raise']. Defaults to 'raise'
              - rows: scales the columns totals so that their sums matches the row totals
              - columns: scales the row totals so that their sums matches the colum totals
              - average: scales both row and column totals to the average value of their sums
              - raise: raises an Exception if the sums of the row and column totals do not match
        max_iterations (int, optional): Maximum number of iterations, defaults to 1000
        rel_error (float, optional): Relative error stopping criteria, defaults to 10e-5
        n_procs (int, optional): Number of processors for parallel computation. Defaults to 1.

    Return:
        Numpy ndarray(M, M): balanced matrix
        float: residual
        int: n_iterations
    """
    max_iterations = int(max_iterations)
    n_procs = int(n_procs)

    # Test if matrix is Pandas DataFrame
    data_type = ''
    if isinstance(m, _pd.DataFrame):
        data_type = 'pd'
        m_pd = m
        m = m_pd.values

    if isinstance(a, _pd.Series) or isinstance(a, _pd.DataFrame):
        a = a.values
    if isinstance(b, _pd.Series) or isinstance(b, _pd.DataFrame):
        b = b.values

    # ##################################################################################
    # Validations:
    #   - m is an MxM square matrix, a and b are vectors of size M
    #   - totals_to_use is one of ['rows', 'columns', 'average']
    #   - the max_iterations is a +'ve integer
    #   - rel_error is a +'ve float between 0 and 1
    #   - the n_procs is a +'ve integer between 1 and the number of available processors
    # ##################################################################################
    valid_totals_to_use = ['rows', 'columns', 'average', 'raise']
    assert m.ndim == 2 and m.shape[0] == m.shape[1], "m must be a two-dimensional square matrix"
    assert a.ndim == 1 and a.shape[0] == m.shape[0], \
        "'a' must be a one-dimensional array, whose size matches that of 'm'"
    assert b.ndim == 1 and b.shape[0] == m.shape[0], \
        "'a' must be a one-dimensional array, whose size matches that of 'm'"
    assert totals_to_use in valid_totals_to_use, "totals_to_use must be one of %s" % valid_totals_to_use
    assert max_iterations >= 1, "max_iterations must be integer >= 1"
    assert 0 < rel_error < 1.0, "rel_error must be float between 0.0 and 1.0"
    assert 1 <= n_procs <= _mp.cpu_count(), \
        "n_procs must be integer between 1 and the number of processors (%d) " % _mp.cpu_count()
    if n_procs > 1:
        raise NotImplementedError("Multiprocessing capability is not implemented yet.")

    # Scale row and column totals, if required
    a_sum = a.sum()
    b_sum = b.sum()
    if not _np.isclose(a_sum, b_sum):
        if totals_to_use == 'rows':
            b = _np.multiply(b, a_sum / b_sum)
        elif totals_to_use == 'columns':
            a = _np.multiply(a, b_sum / a_sum)
        elif totals_to_use == 'average':
            avg_sum = 0.5 * (a_sum + b_sum)
            a = _np.multiply(a, avg_sum / a_sum)
            b = _np.multiply(b, avg_sum / b_sum)
        else:
            raise RuntimeError("a and b vector totals do not match.")

    initial_error = _calc_error(m, a, b)
    err = 1.0
    i = 0
    while i < max_iterations and err > rel_error:
        m = _balance(m, a, 1)
        m = _balance(m, b, 0)
        err = _calc_error(m, a, b) / initial_error
        i += 1

    if data_type == 'pd':
        new_df = _pd.DataFrame(m, index=m_pd.index, columns=m_pd.columns)
        return new_df, err, i
    else:
        return m, err, i


def _balance(matrix, tot, axis):
    """ Balances a matrix using a single constraint.

    Args:
        matrix (numpy ndarray: Matrix to be balanced
        tot (numpy ndarray): Totals
        axis (int): Direction to constrain (0 = along columns, 1 = along rows)

    Return:
        w :  Numpy ndarray(..., M, M)
    """
    sc = tot / matrix.sum(axis)
    sc = _np.nan_to_num(sc)  # replace divide by 0 errors from the prev. line
    if axis:  # along rows
        matrix = _np.multiply(matrix.T, sc).T
    else:   # along columns
        matrix = _np.multiply(matrix, sc)
    return matrix


def _calc_error(m, a, b):
    row_sum = _np.absolute(a - m.sum(1)).sum()
    col_sum = _np.absolute(b - m.sum(0)).sum()
    return row_sum + col_sum