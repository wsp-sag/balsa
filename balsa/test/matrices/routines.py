import unittest

import numpy as np
import pandas as pd

from balsa.matrices.routines import matrix_bucket_rounding

if __name__ == '__main__':
    unittest.main()


class TestMatrixBucketRounding(unittest.TestCase):

    def test_small(self):
        """ Test bucket rounding routine on a small matrix to various levels of rounding precision. """
        a = np.random.uniform(0, 1000, (5, 5))
        for decimal in range(-2, 6):
            a_rnd = matrix_bucket_rounding(a, decimal)
            self._compare_matrix_sums(a_rnd, a, decimal)
            self._compare_matrix_values(a_rnd, a, decimal)

    def test_return_type(self):
        """ Test that bucket rounding returns an integer or float dtype, where appropriate. """
        a = np.random.uniform(0, 1000, (5, 5))

        # first test, float return
        b = matrix_bucket_rounding(a, decimals=2)
        assert b.dtype == a.dtype
        # second test, int return
        b = matrix_bucket_rounding(a, decimals=0)
        assert b.dtype == np.dtype('int32')

    def test_large(self):
        """ Test bucket rounding routine on a large matrix to various levels of rounding precision. """
        a = np.random.uniform(0, 1000, (1000, 1000))
        for decimal in [-2, 0, 5]:
            a_rnd = matrix_bucket_rounding(a, decimal)
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
        assert type(df_rnd) == pd.DataFrame

    def _compare_matrix_sums(self, a, b, decimal):
        max_error = 0.5*(10.0 ** (-decimal))
        a_sum = np.sum(a)
        b_sum = np.sum(b)
        self.assertLessEqual(a_sum, b_sum + max_error)
        self.assertGreaterEqual(a_sum, b_sum - max_error)

    def _compare_matrix_values(self, a, b, decimal):
        max_error = 10.0 ** (-decimal)
        np.testing.assert_allclose(a, b, atol=max_error, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
