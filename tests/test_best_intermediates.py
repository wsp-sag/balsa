import unittest

import numpy as np
import pandas as pd
from pandas import testing as pdt

from wsp_balsa.routines.best_intermediates import (
    best_intermediate_subset_zones, best_intermediate_zones)


class TestBestIntermediates(unittest.TestCase):

    def setUp(self):
        normal_zones = [1, 2]
        intermediate_zones = [100, 101, 102, 103]

        self.pk_index = pd.MultiIndex.from_product([normal_zones, intermediate_zones], names=['p', 'k'])
        self.kq_index = pd.MultiIndex.from_product([intermediate_zones, normal_zones], names=['k', 'q'])
        self.pq_index = pd.MultiIndex.from_product([normal_zones, normal_zones], names=['p', 'q'])

        self.pk_distance = pd.Series(
            [5.093, 1.8829, 3.1976, 0.7403, 4.6856, 7.8176, 3.2228, 5.2814], index=self.pk_index
        )

        self.pk_table = pd.DataFrame({
            'util': [-0.1569, -0.4972, -0.4174, -0.1145, -0.1197, -0.0175, -0.3938, -0.9686],
            'col1': range(8),
            'col2': range(8)
        }, index=self.pk_index)

        self.kq_table = pd.DataFrame({
            'util': [-0.739, -np.inf, -np.inf, -0.3384, -0.9853, -0.1971, -np.inf, -0.896],
            'col1': range(8),
            'col3': range(8)
        }, index=self.kq_index)

    def test_best_path(self):
        expected_result = pd.DataFrame([
            [100, True, -0.8959, 0, 0, 0],
            [102, True, -0.6145, 7, 2, 5],
            [100, True, -0.8587, 4, 4, 0],
            [101, True, -0.3559, 8, 5, 3],
        ], index=self.pq_index, columns=['intermediate_zone', 'available', 'util', 'col1', 'col2', 'col3'])

        df = best_intermediate_zones(self.pk_table, self.kq_table, 'util')

        pdt.assert_frame_equal(expected_result, df)

    def test_best_subset_path(self):
        expected_result = pd.DataFrame([
            [100, True, -0.8959, 0, 0, 0],
            [101, True, -0.8356, 4, 1, 3],
            [100, True, -0.8587, 4, 4, 0],
            [102, True, -0.5909, 11, 6, 5],
        ], index=self.pq_index, columns=['intermediate_zone', 'available', 'util', 'col1', 'col2', 'col3'])

        df = best_intermediate_subset_zones(
            self.pk_distance, self.pk_table, self.kq_table, 'util', n_subset=2, maximize_subset=False
        )

        pdt.assert_frame_equal(expected_result, df)

    # TODO: More comprehensive tests


if __name__ == '__main__':
    unittest.main()
