import unittest
from bisect import bisect_right

import numpy as np
from numpy.testing import assert_allclose

from balsa.cheval.core import (sample_multinomial_worker, sample_nested_worker, stochastic_multinomial_worker,
                               nested_probabilities, logarithmic_search)
from balsa.cheval.tree import ChoiceTree


class RandomStateProxy(object):
    def __init__(self, fixed_draws):
        self.data = np.array(fixed_draws, dtype=np.float64)

    def uniform(self, shape, *args, **kwargs):
        n = np.prod(shape)
        assert n < len(self.data)

        ret = self.data[: n]
        ret.shape = shape

        return ret


class TestCore(unittest.TestCase):

    def _get_utility_row(self):
        ret = np.array(
            [1.678, 1.689, 1.348, 0.903, 1.845, 0.877, 0.704, 0.482],
            dtype=np.float64
        )
        n = len(ret)
        ret.shape = 1, n

        return ret

    def _get_flattened_tree(self):
        tree = ChoiceTree(None)
        a1 = tree.add_node('A1', 0.3)
        a2 = tree.add_node('A2', 0.6)

        a1.add_node('A3')
        a1.add_node('A4')

        a2.add_node('A5')
        a6 = a2.add_node('A6', 0.5)

        a6.add_node('A7')
        a6.add_node('A8')

        return tree.flatten()

    def test_multinomial_probabilities(self):
        utilities = self._get_utility_row()

        expected_result = np.array(
            [0.181775432, 0.183785999, 0.130682672, 0.083744629, 0.214813892, 0.08159533, 0.068632901, 0.054969145],
            dtype=np.float64
        )
        n = len(expected_result)
        expected_result.shape = 1, n

        test_result = np.zeros(utilities.shape)
        stochastic_multinomial_worker(utilities, test_result)

        assert_allclose(test_result, expected_result)

    def test_nested_probabilities(self):
        utilities = np.float64([0, -1.63364, -0.73754, -0.05488, -0.66127, -1.17165, 0, 0])

        expected_result = np.float64([0, 0.06527, 0.17893, 0.47448, 0.23791, 0.04341, 0, 0])

        tree = ChoiceTree(None)
        auto = tree.add_node("Auto", logsum_scale=0.7)
        auto.add_node("Carpool")
        auto.add_node("Drive")
        pt = tree.add_node("Transit", logsum_scale=0.7)
        pt.add_node("Bus")
        train = pt.add_node("Train", logsum_scale=0.3)
        train.add_node("T Access Drive")
        train.add_node("T Access Walk")

        hierarchy, levels, logsum_scales = tree.flatten()
        test_result = nested_probabilities(utilities, hierarchy, levels, logsum_scales)

        assert_allclose(test_result, expected_result, rtol=1e-4)

    def test_multinomial_sampling(self):
        utilities = self._get_utility_row()

        expected_samples = [
            (0.0, 0),
            (0.1, 0),
            (0.2, 1),
            (0.4, 2),
            (0.5, 3),
            (0.6, 4),
            (0.8, 5),
            (0.9, 6),
            (0.98, 7),
            (1.0, 7)
        ]
        for row_number, (random_draw, expected_index) in enumerate(expected_samples):
            random_numbers = np.array([[random_draw]], dtype=np.float64)
            test_result = np.zeros([1, 1], dtype=np.int64)
            sample_multinomial_worker(utilities, random_numbers, test_result)

            assert test_result[0, 0] == expected_index, "Bad result for row %s" % row_number

    def test_nested_sampling(self):

        utilities = np.float64([[0, -1.63364, -0.73754, -0.05488, -0.66127, -1.17165, 0, 0]])

        tree = ChoiceTree(None)
        auto = tree.add_node("Auto", logsum_scale=0.7)
        auto.add_node("Carpool")
        auto.add_node("Drive")
        pt = tree.add_node("Transit", logsum_scale=0.7)
        pt.add_node("Bus")
        train = pt.add_node("Train", logsum_scale=0.3)
        train.add_node("T Access Drive")
        train.add_node("T Access Walk")

        hierarchy, levels, logsum_scales = tree.flatten()

        expected_samples = [
            (0, 1),
            (0.04, 1),
            (0.2, 2),
            (0.5, 3),
            (0.75, 4),
            (0.99, 5)
        ]

        for row_number, (random_draw, expected_index) in enumerate(expected_samples):
            random_numbers = np.array([[random_draw]], dtype=np.float64)
            test_result = np.zeros([1, 1], dtype=np.int64)

            sample_nested_worker(utilities, random_numbers, hierarchy, levels, logsum_scales, test_result)
            result_cell = test_result[0, 0]

            assert result_cell == expected_index, "Bad result for row %s. Expected %s got %s" % (row_number, expected_index, result_cell)

    def test_bisection_search(self):
        cumsums = np.array([0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0], dtype=np.float64)

        expected_samples = [
            (0.0, 2),
            (0.2, 2),
            (0.4, 7),
            (0.6, 8),
            (0.8, 9),
            (0.99, 9)
        ]

        for random_draw, expected_index in expected_samples:
            test_result = logarithmic_search(np.float64(random_draw), cumsums)
            assert test_result == expected_index

            standard_result = bisect_right(cumsums, random_draw)
            assert test_result == standard_result


if __name__ == '__main__':
    unittest.main()
