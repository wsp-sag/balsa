from .core import INSTRUCTION_TYPE_1, INSTRUCTION_TYPE_2
from six import iterkeys, itervalues

import numpy as np
import pandas as pd


class _ChoiceNode(object):

    def __init__(self, *args):
        self._name, self._root, self._parent, self.logsum_scale, self._level = args
        self._children = set()

    def __str__(self):
        return self._name

    def __repr__(self):
        return "ChoiceNode(%s)" % self._name

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

    def max_level(self):
        max_level = self._level

        for c in self.children():
            max_level = max(max_level, c.max_level())

        return max_level

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


class ChoiceTree(object):

    def __init__(self, root):
        self._root = root

        self._all_nodes = {}
        self._children = set()
        self._cached_node_index = None

    def __getitem__(self, item):
        return self._all_nodes[item]

    def max_level(self):
        max_level = 1

        for c in self.children():
            max_level = max(max_level, c.max_level())

        return max_level

    def children(self):
        for c in self._children:
            yield c

    @property
    def node_index(self):
        if self._cached_node_index is None:
            idx = pd.Index(sorted(iterkeys(self._all_nodes)))
            self._cached_node_index = idx
            return idx
        return self._cached_node_index

    def _root_add(self, parent, new_name, logsum_scale, level):
        if new_name in self._all_nodes:
            old_node = self._all_nodes[new_name]
            old_node.parent._children.remove(old_node)
        new_node = _ChoiceNode(new_name, self, parent, logsum_scale, level)
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

    def flatten(self):
        """Creates instruction sets for nested models"""
        node_index = self.node_index
        node_positions = {name: i for i, name in enumerate(node_index)}

        # 1. Organize parent nodes by level
        levels = {}
        for node in itervalues(self._all_nodes):
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

