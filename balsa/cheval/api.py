import numpy as np
import pandas as pd
from threading import Thread

from .scope import Scope, ExpressionContainer
from .tree import ChoiceTree
from .core import sample_multinomial_worker, sample_nested_worker, stochastic_multinomial_worker, stochastic_nested_worker


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
        self._check_model_is_ready(compute_utilities=override_utilities is None)

        if randomizer is None:
            randomizer = np.random
        elif isinstance(randomizer, (int, np.int_)):
            randomizer = np.random.RandomState(randomizer)

        assert n_draws >= 1

        if override_utilities is None:
            utilities = self._scope_container._compute_utilities(n_threads, logger=logger)
        else:
            utilities = override_utilities

        result_indices = self._eval_probabilities_and_sample(utilities, randomizer, n_draws, n_threads)

        return self._convert_result(result_indices, astype, squeeze)

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
            logger (None or Logger): Provide a Logger object to record errors during expression evaluation.

        Returns:
            DataFrame of probabilities of each record x each alternative.

        """
        self._check_model_is_ready(compute_utilities=override_utilities is None)

        if override_utilities is None:
            utilities = self._scope_container._compute_utilities(n_threads, logger=logger)
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

    def _check_model_is_ready(self, compute_utilities=True):

        assert len(self._tree_container.node_index) > 1, "At least two choices are required for a model to be valid"

        if compute_utilities:
            assert len(self._expression_container) > 0, "Must define at least one utility expression"
            assert len(self._scope_container._empty_symbols) == 0, "Not all scope symbols have been filled with data"

    def _eval_probabilities_and_sample(self, utilities, randomizer: np.random.RandomState, n_draws, n_threads):

        # TODO: Try out saving the results in the random draws array to save on memory.
        result_shape = utilities.shape[0], n_draws
        random_draws = randomizer.uniform(size=result_shape)
        result = np.zeros(result_shape, dtype=np.int64)

        utility_chunks = np.array_split(utilities, n_threads, axis=0)
        random_chunks = np.array_split(random_draws, n_threads, axis=0)
        result_chunks = np.array_split(result, n_threads, axis=0)

        if self.tree.max_level() == 1:
            # Multinomial model

            threads = [
                Thread(target=sample_multinomial_worker, args=[
                    utility_chunks[i], random_chunks[i], result_chunks[i]
                ])
                for i in range(n_threads)
            ]
        else:
            # Nested model

            instructions1, instructions2 = self.tree.flatten()

            threads = [
                Thread(target=sample_nested_worker, args=[
                    utility_chunks[i], random_chunks[i], instructions1, instructions2, result_chunks[i]
                ])
                for i in range(n_threads)
            ]

        for t in threads: t.start()
        for t in threads: t.join()

        return result

    def _convert_result(self, results, astype, squeeze):
        """
        Takes the discrete outcomes as an ndarray and converts it to a Series or DataFrame of the user-specified type.
        """

        n_draws = results.shape[1]
        record_index = self._scope_container._records
        column_index = pd.Index(range(n_draws))

        if astype == 'index':
            if squeeze and n_draws == 1:
                return pd.Series(results[:, 0], index=record_index)
            return pd.DataFrame(results, index=record_index, columns=column_index)
        elif astype == 'category':
            lookup_table = pd.Categorical(self._tree_container.node_index)
        else:
            lookup_table = self._tree_container.node_index.values.astype(astype)

        retval = []
        for col in range(n_draws):
            indices = results[:, col]
            retval.append(pd.Series(lookup_table.take(indices), index=record_index))
        retval = pd.concat(retval)
        retval.columns = column_index

        return retval

    def _eval_probabilities_only(self, utilities, n_threads):
        result = np.zeros(shape=utilities.shape, dtype=np.float64, order='C')

        utility_chunks = np.array_split(utilities, n_threads, axis=0)
        result_chunks = np.array_split(result, n_threads, axis=0)

        if self.tree.max_level() == 1:
            # Multinomial model

            threads = [
                Thread(target=stochastic_multinomial_worker, args=[
                    utility_chunks[i], result_chunks[i]
                ])
                for i in range(n_threads)
            ]
        else:
            # Nested model

            instructions1, instructions2 = self.tree.flatten()

            threads = [
                Thread(target=stochastic_nested_worker, args=[
                    utility_chunks[i], instructions1, instructions2, result_chunks[i]
                ])
                for i in range(n_threads)
            ]

        for t in threads: t.start()
        for t in threads: t.join()

        return result
