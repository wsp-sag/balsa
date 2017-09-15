import unittest

import pandas as pd
import numpy as np

from balsa.cheval.api import sample_from_weights, ChoiceTree


class RandomStateProxy(object):
    def __init__(self, fixed_draws):
        self.data = np.array(fixed_draws, dtype=np.float64)

    def uniform(self, size, *args, **kwargs):
        n = np.prod(size)
        assert n <= len(self.data)

        ret = self.data[: n]
        ret.shape = size

        return ret


class TestAPI(unittest.TestCase):

    def test_weighted_probabilities(self):
        weights = [
            [0, 0, 1, 0, 0, 2, 3, 0, 0, 1],
            [0, 0, 1, 0, 0, 2, 3, 0, 0, 1],
            [0, 0, 1, 0, 0, 2, 3, 0, 0, 1],
            [0, 0, 1, 0, 0, 2, 3, 0, 0, 1],
            [0, 0, 1, 0, 0, 2, 3, 0, 0, 1],
            [0, 0, 1, 0, 0, 2, 3, 0, 0, 1]
        ]
        weights = pd.DataFrame(weights, index=[101, 102, 103, 104, 105, 106], columns="A B C D E F G H I J".split())
        randomizer = RandomStateProxy([0, 0.05, 0.2, 0.5, 0.8, 0.9])

        expected_result = [2, 2, 5, 6, 6, 9]

        actual_result = sample_from_weights(weights, randomizer, astype='index').values
        assert np.all(actual_result == expected_result)

    def test_flatten(self):
        tree = ChoiceTree(None)
        a1 = tree.add_node('A1', 0.3)
        a2 = tree.add_node('A2', 0.6)

        a1.add_node('A3')
        a1.add_node('A4')

        a2.add_node('A5')
        a6 = a2.add_node('A6', 0.5)

        a6.add_node('A7')
        a6.add_node('A8')

        hierarchy, levels, ls_scales = tree.flatten()

        expected_hierarchy = np.int64([-1, -1, 0, 0, 1, 1, 5, 5])
        expected_levels = np.int64([0, 0, 1, 1, 1, 1, 2, 2])
        expected_scales = np.float64([0.3, 0.6, 1, 1, 1, 0.5, 1, 1])

        assert np.all(expected_hierarchy == hierarchy)
        assert np.all(expected_levels == levels)
        assert np.allclose(expected_scales, ls_scales)


if __name__ == '__main__':
    unittest.main()
