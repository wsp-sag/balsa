import unittest
import numpy as np
from numpy.testing import assert_allclose

from balsa.cheval.core import sample_multinomial_worker, sample_nested_worker, stochastic_multinomial_worker, stochastic_nested_worker
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
        utilities = self._get_utility_row()

        expected_result = np.array(
            [0, 0, 0.171971982, 0.110203821, 0.354475969, 0, 0.201757526, 0.161590702],
            dtype=np.float64
        )
        n = len(expected_result)
        expected_result.shape = 1, n

        instructions1, instructions2 = self._get_flattened_tree()

        test_result = np.zeros(utilities.shape)
        stochastic_nested_worker(utilities, instructions1, instructions2, test_result)

        assert_allclose(test_result, expected_result)

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
        utilities = self._get_utility_row()

        expected_samples = [
            (0, 2),
            (0.1, 2),
            (0.2, 3),
            (0.6, 4),
            (0.7, 6),
            (0.9, 7)
        ]

        instructions1, instructions2 = self._get_flattened_tree()

        for row_number, (random_draw, expected_index) in enumerate(expected_samples):
            random_numbers = np.array([[random_draw]], dtype=np.float64)
            test_result = np.zeros([1, 1], dtype=np.int64)
            sample_nested_worker(utilities, random_numbers, instructions1, instructions2, test_result)

            assert test_result[0, 0] == expected_index, "Bad result for row %s" % row_number


if __name__ == '__main__':
    unittest.main()
