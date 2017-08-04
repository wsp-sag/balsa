import unittest

import pandas as pd
import numpy as np

from balsa.cheval.api import sample_from_weights


class RandomStateProxy(object):
    def __init__(self, fixed_draws):
        self.data = np.array(fixed_draws, dtype=np.float64)

    def uniform(self, shape, *args, **kwargs):
        n = np.prod(shape)
        assert n <= len(self.data)

        ret = self.data[: n]
        ret.shape = shape

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


if __name__ == '__main__':
    unittest.main()
