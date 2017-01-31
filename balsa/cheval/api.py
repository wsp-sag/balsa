import numpy as np
import pandas as pd

from .scope import Scope, ExpressionContainer
from .tree import ChoiceTree


class ChoiceModel(object):

    def __init__(self):
        self._expression_container = ExpressionContainer(self)
        self._scope_container = Scope(self)
        self._tree_container = ChoiceTree(self)

    @property
    def scope(self):
        return self._scope_container

    @property
    def expressions(self):
        return self._expression_container

    @property
    def tree(self):
        return self._tree_container

    def run_discrete(self, randomizer=None, n_draws=1, astype='category', squeeze=True, n_threads=1,
                     override_utilities=None, logger=None):
        """
        For each record, discretely sample one or more times (with replacement) from the probability distribution.

        Args:
            randomizer (RandomState, int, or None): If a RandomState instance is given, it will be used to generate
                random draws for the model. If int is provided, it will be passed as the seed to new RandomState. If
                None, numpy.random will be used instead of a consistent RandomState.
            n_draws (int): The number of times to draw (with replacement) for each record. Must be >= 1. Run time is
                proportional to the number of draws.
            astype (dtype or 'category' or 'index'): The dtype of the return array; the result will be cast to the
                given dtype. The special value 'category' returns a Categorical Series (or a DataFrame for n_draws > 1).
                The special value 'index' returns the positional index in the sorted array of node names.
            squeeze (bool): Only used when n_draws == 1. If True, then a Series will be returned, otherwise a DataFrame
                with one column will be returned.
            n_threads (int): The number of threads to uses in the computation. Must be >= 1
            override_utilities (None or DataFrame): If not None, then the model will assume a pre-computed set of
                utilities for each record x each alternative; otherwise the built-in utility computation framework will
                be used. The columns of the given DataFrame MUST match the sorted list of node (alternative) names.
            logger (None or Logger): Optionally provide a Logger to report progress during the run. Progress will be
                reported at the INFO level.

        Returns:
            Series or DataFrame, depending on squeeze and n_draws. The dtype of the returned object depends on astype.

        """
        self._check_model_is_ready()

        if randomizer is None:
            randomizer = np.random
        elif isinstance(randomizer, (int, np.int_)):
            randomizer = np.random.RandomState(randomizer)

        assert n_draws >= 1

        if override_utilities is None:
            utilities = self._eval_utilities(n_threads)
        else:
            utilities = override_utilities

        result_indices = self._eval_probabilities_and_sample(utilities, randomizer, n_draws, n_threads)

        return self._convert_result(result_indices, n_draws, astype, squeeze)

    def run_stochastic(self, n_threads=1, override_utilities=None, logger=None):
        """
        For each record, compute the probability distribution of the logit model. A DataFrame will be returned whose
        columns match the sorted list of node names (alternatives) in the model. Probabilities over all alternatives for
        each record will sum to 1.0.

        Args:
            n_threads (int): The number of threads to be used in the computation. Must be >= 1.
            override_utilities (None or DataFrame): If not None, then the model will assume a pre-computed set of
                utilities for each record x each alternative; otherwise the built-in utility computation framework will
                be used. The columns of the given DataFrame MUST match the sorted list of node (alternative) names.
            logger (None or Logger): Optionally provide a Logger to report on progress during the run. Progress will be
                reported at the INFO level.

        Returns:
            DataFrame of probabilities of each record x each alternative.

        """
        self._check_model_is_ready()

        if override_utilities is None:
            utilities = self._eval_utilities(n_threads)
        else:
            utilities = override_utilities

        results = self._eval_probabilities_only(utilities, n_threads)

        return results

    def copy(self, expressions=False, scope=False):
        """
        Creates a copy of this model's structure, optionally copying the scope and expressions.

        Args:
            expressions (bool):
            scope:

        Returns:

        """
        raise NotImplementedError()

    def _check_model_is_ready(self):
        raise NotImplementedError()

    def _eval_utilities(self, n_threads):
        raise NotImplementedError()

    def _eval_probabilities_and_sample(self, utilities, randomizer, n_draws, n_threads):
        raise NotImplementedError()

    def _eval_probabilities_only(self, utilities, n_threads):
        raise NotImplementedError()

    def _convert_result(self, results, n_draws, astype, squeeze):
        raise NotImplementedError()
