import unittest

import numpy as np
import pandas as pd
from pandas import testing as pdt

from wsp_balsa.routines import (aggregate_matrix, matrix_balancing_1d,
                                matrix_balancing_2d, matrix_bucket_rounding)


class TestMatrixBucketRounding(unittest.TestCase):

    def test_small(self):
        """ Test bucket rounding routine on a small matrix to various levels of rounding precision. """
        a = np.random.uniform(0, 1000, (5, 5))
        for decimal in range(-2, 6):
            a_rnd = matrix_bucket_rounding(a, decimals=decimal)

            self._compare_matrix_sums(a_rnd, a, decimal)
            self._compare_matrix_values(a_rnd, a, decimal)

    def test_return_type(self):
        """ Test that bucket rounding returns an integer or float dtype, where appropriate. """
        a = np.random.uniform(0, 1000, (5, 5))

        # first test, float return
        b = matrix_bucket_rounding(a, decimals=2)
        self.assertEqual(b.dtype, a.dtype, "dtype of bucket rounded matrix is not equal to dtype of input matrix")

        # second test, int return
        b = matrix_bucket_rounding(a, decimals=0)
        self.assertEqual(b.dtype, np.dtype('int32'), "dtype of bucket rounded matrix is not integer")

    def test_large(self):
        """ Test bucket rounding routine on a large matrix to various levels of rounding precision. """
        a = np.random.uniform(0, 1000, (1000, 1000))
        for decimal in [-2, 0, 5]:
            a_rnd = matrix_bucket_rounding(a, decimals=decimal)

            self._compare_matrix_sums(a_rnd, a, decimal)
            self._compare_matrix_values(a_rnd, a, decimal)

    def test_pandas_import(self):
        """Test return type and values if matrix is passed as a Pandas DataFrame. """
        a = np.random.uniform(0, 1000, (5, 5))
        df = pd.DataFrame(a)
        decimals = 3
        df_rnd = matrix_bucket_rounding(df, decimals=decimals)

        self._compare_matrix_sums(df.values, df_rnd.values, decimals)
        self._compare_matrix_values(df.values, df_rnd.values, decimals)
        self.assertEqual(type(df_rnd), pd.DataFrame, "dtype of returned matrix is a Pandas DataFrame")

    def _compare_matrix_sums(self, a, b, decimal):
        max_error = 0.5 * (10.0 ** (-decimal))
        a_sum = np.sum(a)
        b_sum = np.sum(b)

        self.assertLessEqual(a_sum, b_sum + max_error, "Bucket rounded matrix is not within a small margin of error")
        self.assertGreaterEqual(a_sum, b_sum - max_error, "Bucket rounded matrix is not within a small margin of error")

    def _compare_matrix_values(self, a, b, decimal):
        max_error = 10.0 ** (-decimal)
        np.testing.assert_allclose(a, b, atol=max_error, rtol=0.0,
                                   err_msg="Bucket rounded matrix values are not within %f" % (max_error))


class TestAggregateMatrix(unittest.TestCase):

    def setUp(self):
        self._square_symatrix = np.array([
            [4, 9, 1, 1, 9, 0, 4, 4, 9],
            [10, 0, 0, 10, 2, 8, 0, 10, 9],
            [8, 9, 9, 6, 7, 7, 4, 1, 8],
            [9, 8, 1, 10, 6, 7, 2, 1, 2],
            [3, 3, 10, 5, 3, 9, 7, 9, 4],
            [1, 5, 1, 1, 7, 4, 2, 9, 0],
            [4, 3, 4, 1, 5, 9, 3, 7, 5],
            [2, 3, 7, 2, 2, 10, 2, 3, 5],
            [5, 10, 4, 9, 1, 5, 4, 4, 7]
        ])
        self._numeric_index = pd.Index([10, 11, 12, 20, 21, 22, 30, 31, 32])
        self._square_symatrix = pd.DataFrame(self._square_symatrix, index=self._numeric_index, columns=self._numeric_index)
        self._tall_symatrix = self._square_symatrix.stack()

        self._square_dmatrix = np.array([
            [1, 3, 2, 9],
            [5, 0, 4, 7],
            [10, 5, 4, 0],
            [6, 1, 8, 9],
            [10, 5, 0, 3],
            [4, 8, 8, 6],
            [1, 0, 4, 1],
            [2, 3, 4, 6],
            [3, 0, 0, 7],
        ])
        self._text_index = pd.Index(['A1', 'A2', 'B1', 'B2'])
        self._square_dmatrix = pd.DataFrame(self._square_dmatrix, index=self._numeric_index, columns=self._text_index)
        self._tall_dmatrix = self._square_dmatrix.stack()

        self._grouper1 = pd.Series({
            10: 1,
            11: 1,
            12: 1,
            20: 2,
            21: 2,
            22: 2,
            30: 3,
            31: 3,
            32: 3
        })

        self._grouper2 = pd.Series({
            'A1': 'A',
            'A2': 'A',
            'B1': 'B',
            'B2': 'B'
        })

    def test_square_symatrix(self):
        expected_result = pd.DataFrame([
            [50, 50, 49],
            [41, 52, 36],
            [42, 44, 40]
        ], index=[1, 2, 3], columns=[1, 2, 3])

        test1 = aggregate_matrix(self._square_symatrix, row_groups=self._grouper1, col_groups=self._grouper1)
        pdt.assert_frame_equal(expected_result, test1, check_dtype=False)

        test2 = aggregate_matrix(self._square_symatrix, groups=self._grouper1)
        pdt.assert_frame_equal(expected_result, test2, check_dtype=False)

        test3 = aggregate_matrix(self._square_symatrix, groups=self._grouper1.values)
        pdt.assert_frame_equal(expected_result, test3, check_dtype=False)

    def test_square_dmatrix(self):
        expected_result = pd.DataFrame([
            [24, 26],
            [34, 34],
            [9, 22],
        ], index=[1, 2, 3], columns=['A', 'B'])

        test1 = aggregate_matrix(self._square_dmatrix, row_groups=self._grouper1, col_groups=self._grouper2)
        pdt.assert_frame_equal(expected_result, test1, check_dtype=False)

        test2 = aggregate_matrix(self._square_dmatrix, row_groups=self._grouper1.values,
                                 col_groups=self._grouper2.values)
        pdt.assert_frame_equal(expected_result, test2, check_dtype=False)

    def test_tall_symatrix(self):
        expected_result = pd.DataFrame([
            [1, 1, 50],
            [1, 2, 50],
            [1, 3, 49],
            [2, 1, 41],
            [2, 2, 52],
            [2, 3, 36],
            [3, 1, 42],
            [3, 2, 44],
            [3, 3, 40],
        ], columns=['row', 'col', 'val']).set_index(['row', 'col'])['val']

        tall_row_grouper = self._grouper1.reindex(self._tall_symatrix.index, level=0)
        tall_col_grouper = self._grouper1.reindex(self._tall_symatrix.index, level=1)

        test1 = aggregate_matrix(self._tall_symatrix, row_groups=tall_row_grouper, col_groups=tall_col_grouper)
        pdt.assert_series_equal(expected_result, test1, check_dtype=False, check_names=False)

        test2 = aggregate_matrix(self._tall_symatrix, row_groups=self._grouper1, col_groups=self._grouper1)
        pdt.assert_series_equal(expected_result, test2, check_dtype=False, check_names=False)

        test3 = aggregate_matrix(self._tall_symatrix,
                                 row_groups=tall_row_grouper.values, col_groups=tall_col_grouper.values)
        pdt.assert_series_equal(expected_result, test3, check_dtype=False, check_names=False)

        test4 = aggregate_matrix(self._tall_symatrix, groups=self._grouper1)
        pdt.assert_series_equal(expected_result, test4, check_dtype=False, check_names=False)

    def test_tall_dmatrix(self):
        expected_result = pd.DataFrame([
            [1, 'A', 24],
            [1, 'B', 26],
            [2, 'A', 34],
            [2, 'B', 34],
            [3, 'A', 9],
            [3, 'B', 22]
        ], columns=['row', 'col', 'val']).set_index(['row', 'col'])['val']

        tall_row_grouper = self._grouper1.reindex(self._tall_dmatrix.index, level=0)
        tall_col_grouper = self._grouper2.reindex(self._tall_dmatrix.index, level=1)

        test1 = aggregate_matrix(self._tall_dmatrix, row_groups=tall_row_grouper, col_groups=tall_col_grouper)
        pdt.assert_series_equal(expected_result, test1, check_dtype=False, check_names=False)

        test2 = aggregate_matrix(self._tall_dmatrix, row_groups=self._grouper1, col_groups=self._grouper2)
        pdt.assert_series_equal(expected_result, test2, check_dtype=False, check_names=False)

        test3 = aggregate_matrix(self._tall_dmatrix,
                                 row_groups=tall_row_grouper.values, col_groups=tall_col_grouper.values)
        pdt.assert_series_equal(expected_result, test3, check_dtype=False, check_names=False)


class TestMatrixBalancing(unittest.TestCase):
    def setUp(self):
        self._square_matrix = np.random.uniform(0, 1000, (5, 5))
        self._1darray = np.random.uniform(0, 1000, 5)

    def test_1d_balance(self):
        axes = [0, 1]
        for ax in axes:
            test = matrix_balancing_1d(self._square_matrix, self._1darray, ax)
            self.assertAlmostEqual(test.sum(), self._1darray.sum(), places=5)
            pdt.assert_series_equal(pd.Series(np.sum(test, ax)), pd.Series(self._1darray))

    def test_2d_balance_matched_total(self):
        row = self._1darray
        column = np.roll(self._1darray, 2)

        test = matrix_balancing_2d(self._square_matrix, row, column, rel_error=0.000001)
        pdt.assert_series_equal(pd.Series(np.sum(test[0], 1)), pd.Series(row), check_exact=False)
        pdt.assert_series_equal(pd.Series(np.sum(test[0], 0)), pd.Series(column), check_exact=False)

    def test_2d_balance_average_total(self):
        row = self._1darray
        column = np.roll(np.sqrt(row), 2)

        test = matrix_balancing_2d(self._square_matrix, row, column, rel_error=0.000001, totals_to_use='average')
        self.assertAlmostEqual(test[0].sum().sum(), (row.sum() + column.sum())/2, places=5)

    def test_2d_balance_row_total(self):
        row = self._1darray
        column = np.sqrt(row)

        test = matrix_balancing_2d(self._square_matrix, row, column, rel_error=0.000001, totals_to_use='rows')
        pdt.assert_series_equal(pd.Series(np.sum(test[0], 1)), pd.Series(row), check_exact=False)

    def test_2d_balance_col_total(self):
        row = self._1darray
        column = np.sqrt(row)

        test = matrix_balancing_2d(self._square_matrix, row, column, rel_error=0.000001, totals_to_use='columns')
        pdt.assert_series_equal(pd.Series(np.sum(test[0], 0)), pd.Series(column), check_exact=False)


if __name__ == '__main__':
    unittest.main()
