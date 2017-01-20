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

import io


_err_msg_sq_mat = "%s must be a two-dimensional square matrix"
_err_msg_2d_mat = "%s must be a two-dimensional matrix"
_err_msg_vector = "%s must be a one-dimensional array, whose size matches that of %s"
_err_msg_totals_not_in_list = "%s must be one of %s"
_err_msg_not_integer = "%s must be an integer"
_err_msg_not_float = "%s must be a floating point number"
_err_msg_incompatible_shapes = "axis %d of matrice %s and %s must be the same."


def matrix_balancing_1d(m, a, axis):
    """ Balances a matrix using a single constraint.

    Args:
        m (numpy ndarray (M, M)): Matrix to be balanced
        a (numpy ndarray (M)): Totals
        axis (int): Direction to constrain (0 = along columns, 1 = along rows)

    Return:
        w :  Numpy ndarray(..., M, M)
    """

    try:
        axis = int(axis)
        assert axis in [0, 1]
    except:
        raise RuntimeError("axis must be either 0 or 1")

    assert m.ndim == 2, _err_msg_2d_mat % "m"
    assert a.ndim == 1, _err_msg_vector % "a"
    assert m.shape[axis] == a.shape[0], _err_msg_incompatible_shapes % (axis, "m", "a")

    return _balance(m, a, axis)


def matrix_balancing_2d(m, a, b, totals_to_use='raise', max_iterations=10,
                        rel_error=1.0e-5, n_procs=1):
    """ Balances a two-dimensional matrix using iterative proportional fitting.

    Args:
        m (numpy ndarray (M, M): Matrix to be balanced
        a (numpy ndarray (M)): Row totals
        b (numpy ndarray (M)): Column totals
        totals_to_use (str, optional):
            Describes how to scale the row and column totals if their sums do not match
            Must be one of ['rows', 'columns', 'average', 'raise']. Defaults to 'raise'
        max_iterations (int, optional): Maximum number of iterations, defaults to 10
        rel_error (float, optional): Relative error stopping criteria, defaults to 10e-5
        n_procs (int, optional): Number of processors for parallel computation. Defaults to 1.

    Return:
        Numpy ndarray(M, M): balanced matrix
        float: residual
        int: n_iterations
    """

    # ##################################################################################
    # Validations:
    #   - m is an MxM square matrix, a and b are vectors of size M
    #   - totals_to_use is one of ['rows', 'columns', 'average']
    #   - the max_iterations is a +'ve integer
    #   - rel_error is a +'ve float between 0 and 1
    #   - the n_procs is a +'ve integer between 1 and the number of available processors
    # ##################################################################################
    valid_totals_to_use = ['rows', 'columns', 'average', 'raise']
    assert m.ndim == 2 and m.shape[0] == m.shape[1], _err_msg_sq_mat % "m"
    assert a.ndim == 1 and a.shape[0] == m.shape[0], _err_msg_vector % ("a", "m")
    assert b.ndim == 1 and b.shape[0] == m.shape[0], _err_msg_vector % ("a", "m")
    assert totals_to_use in valid_totals_to_use, _err_msg_totals_not_in_list % (
        "totals_to_use", valid_totals_to_use)

    try:
        max_iterations = int(max_iterations)
        assert max_iterations >= 1
    except:
        raise RuntimeError("max_iterations must be integer >= 1")

    try:
        rel_error = float(rel_error)
        assert 0 < rel_error < 1.0
    except:
        raise RuntimeError("rel_error must be float between 0.0 and 1.0")

    try:
        n_procs = int(n_procs)
        assert 1 <= n_procs <= _mp.cpu_count()
    except:
        raise RuntimeError("n_procs must be integer between 1 and "
                           "the number of processors (%d) " % _mp.cpu_count())

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
    if axis:  # along rows
        matrix = _np.multiply(matrix.T, sc).T
    else:   # along columns
        matrix = _np.multiply(matrix, sc)
    return matrix


def _calc_error(m, a, b):
    row_sum = _np.absolute(a - m.sum(1)).sum()
    col_sum = _np.absolute(b - m.sum(0)).sum()
    return row_sum + col_sum
