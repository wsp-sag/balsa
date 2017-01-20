from collections import deque
from .core import INSTRUCTION_TYPE_1, INSTRUCTION_TYPE_2

import pandas as pd
import numpy as np


class _LogitNode(object):

    def __init__(self, *args):
        self._name, self._root, self._parent, self.logsum_scale, self._level = args
        self._children = set()

    def __str__(self):
        return self._name

    def __repr__(self):
        return "LogitNode(%s)" % self._name

    @property
    def name(self):
        return self._name

    @property
    def root(self):
        return self._root

    @property
    def parent(self):
        return self._parent

    @property
    def level(self):
        return self._level

    @property
    def is_parent(self):
        return len(self._children) > 0

    def children(self):
        for c in self._children:
            yield c

    def add_node(self, name, logsum_scale=1.0):
        """
        Adds a nested alternative to the logit model. The name must be unique.

        Args:
            name (str): The name of the alternative in the choice set.
            logsum_scale (int): The logsum scale parameter. Not used in the probability computation if this node has no
                children. A value of 1.0 can be used if the estimated coefficients are already scaled.

        Returns:
            The added node, which also has an 'add_node' method.

        """
        return self._root._root_add(self, name, logsum_scale, self._level + 1)


class LogitModel(object):

    def __init__(self, use_single_precision=False):
        self._all_nodes = {}
        self._children = set()
        self._cached_node_index = None
        self.use_single_precision = bool(use_single_precision)

    def __getitem__(self, item):
        return self._all_nodes[item]

    def children(self):
        for c in self._children:
            yield c

    def _root_add(self, parent, new_name, logsum_scale, level):
        if new_name in self._all_nodes:
            old_node = self._all_nodes[new_name]
            old_node.parent._children.remove(old_node)
        new_node = _LogitNode(new_name, self, parent, logsum_scale, level)
        parent._children.add(new_node)
        self._all_nodes[new_name] = new_node

        # Clear the cached node index because the set of alternatives is being changed
        self._cached_node_index = None
        return new_node

    def add_node(self, name, logsum_scale=1.0):
        """
        Adds a top-level alternative to the logit model. The name must be unique.

        Args:
            name (str): The name of the alternative in the choice set.
            logsum_scale (int): The logsum scale parameter. Not used in the probability computation if this node has no
                children. A value of 1.0 can be used if the estimated coefficients are already scaled.

        Returns:
            The added node, which also has an 'add_node' method.

        """
        return self._root_add(self, name, logsum_scale, 1)

    def remove_node(self, name):
        """
        Removes a node (at any level) from the logit model.

        Args:
            name (str): The name of the node to remove

        """

        old_node = self._all_nodes[name]
        old_node.parent._children.remove(old_node)
        del self._all_nodes[name]

        # Clear the cached node index because the set of alternatives is being changed
        self._cached_node_index = None

    @property
    def scope(self):
        raise NotImplementedError()

    @property
    def expressions(self):
        raise NotImplementedError()

    @property
    def node_index(self):
        if self._cached_node_index is None:
            idx = pd.Index(sorted(self._all_nodes.iterkeys()))
            self._cached_node_index = idx
            return idx
        return self._cached_node_index

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def copy(self, expressions=False, scope=False):
        """
        Creates a copy of this model's structure, optionally copying the scope and expressions.

        Args:
            expressions (bool):
            scope:

        Returns:

        """
        raise NotImplementedError()

    def _flatten(self):
        """Creates instruction sets for nested models"""
        node_index = self.node_index
        node_positions = {name: i for i, name in enumerate(node_index)}

        # 1. Organize parent nodes by level
        levels = {}
        for node in self._all_nodes.itervalues():
            if not node.is_parent: continue
            level = node.level

            if level not in levels: levels[level] = [node]
            else: levels[level].append(node)

        # 2. Construct instruction set 1
        instruction_set_1 = []
        for level in sorted(levels.keys(), reverse=True):
            parent_nodes = levels[level]
            for parent_node in parent_nodes:
                for node in parent_node.children():
                    node_position = node_positions[node.name]
                    instruction_set_1.append((node_position, False, parent_node.logsum_scale))
                parent_position = node_positions[parent_node.name]
                instruction_set_1.append((parent_position, True, parent_node.logsum_scale))
        for node in self.children():
            node_position = node_positions[node.name]
            instruction_set_1.append((node_position, False, 1.0))
        instruction_set_1 = np.rec.array(instruction_set_1, dtype=INSTRUCTION_TYPE_1)

        # 3. Construct instruction set 2
        instruction_set_2 = []
        for level in sorted(levels.keys()):
            parent_nodes = levels[level]
            for parent_node in parent_nodes:
                parent_position = node_positions[parent_node.name]
                for node in parent_node.children():
                    node_position = np.int64(node_positions[node.name])
                    instruction_set_2.append((node_position, parent_position))
        instruction_set_2 = np.rec.array(instruction_set_2, dtype=INSTRUCTION_TYPE_2)

        return instruction_set_1, instruction_set_2
