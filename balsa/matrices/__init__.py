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
from __future__ import print_function as _print_function

import numpy as _np

import io


def matrix_balancing_1d():
    """ Balances a matrix using a single constraint.

    Args:
        m (numpy array_like (M, M): Matrix to be balanced
        a (numpy array_like (M)): Totals
        axis (int): Direction to constrain (0 = rows, 1 = columns)

    Return:
    w :  Numpy ndarray(..., M, M)
    """
    print("To be completed.")


def matrix_balancing_2d():
    """ Balances a two-dimensional matrix using iterative proportional fitting.

    Args:
        m (numpy array_like (M, M): Matrix to be balanced
        a (numpy array_like (M)): Row totals
        b (numpy array_like (M)): Column totals
        totals_to_use (str, optional):
            Describes how to scale the row and column totals if their sums do not match
            Must be one of ['rows', 'columns', 'average']
        max_iterations (int, optional): Maximum number of iterations, defaults to 10
        rel_error (float, optional): Relative error stopping criteria, defaults to 10e-5
        n_procs (int, optional): Number of processors for parallel computation. Defaults to 1.

    Return:
    w :  Numpy ndarray(..., M, M)
    """
    print("To be completed.")