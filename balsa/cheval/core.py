from __future__ import division, absolute_import, print_function
# Note: unicode literals is deliberately NOT imported because it causes an error with np.dtype in Py2

import numba as nb
import numpy as np


'''
These dtypes are required to instruct the Numba machines in what order to process a nested logit model. Numba otherwise
has no way of understanding the the tree (it doesn't permit hierarchical data).

Instruction type 1 is used to compute (from the utilities) the probability of each alternative conditional on its
parent. For example, the particular probability of all child nodes under a particular alternative will sum to 1.0,
but the entire array will not. This part of the computation also applies the logsum scaling parameter. The nature of
this computation is such that probabilities are computed "bottom-up", starting with the lowest level nests and
processing all of the nodes at EACH level (regardless of whether they share a parent node) before moving up a level.

Instruction type 2 is used to compute the absolute probability for the entire tree, working in the opposite direction as
before. The probability of each child node is multiplied by that of its parent, and then each parent node has its
probability zeroed out.
'''
INSTRUCTION_TYPE_1 = np.dtype([('node_index', 'i8'), ('logsum_flag', '?'), ('logsum_scale', 'f8')])
_NB_INSTRUCTION_TYPE_1 = nb.from_dtype(INSTRUCTION_TYPE_1)

INSTRUCTION_TYPE_2 = np.dtype([('child_index', 'i8'), ('parent_index', 'i8')])
_NB_INSTRUCTION_TYPE_2 = nb.from_dtype(INSTRUCTION_TYPE_2)


@nb.jit(nb.int64(nb.float64, nb.float64[:]), nogil=True, nopython=True)
def logarithmic_search(r, cps):
    """
    Logarithmic (binary) search algorithm for finding the greatest index whose cumulative probability is <= the random
    draw.

    Allows for cells with 0 probability.

    Args:
        r (float): The random draw to compare against.
        cps (float[]): The cumulative probabilities to search

    Returns (int): The found index.
    """

    mask, = np.where(cps > 0)  # The masked indices can be used to transform back to the original array
    masked_cps = cps[mask]

    ncols = len(masked_cps)

    lower_bound, upper_bound = 0, ncols - 1
    while (upper_bound - lower_bound) > 1:
        mid_index = np.uint32((upper_bound + lower_bound) // 2)
        cp_at_mid = masked_cps[mid_index]
        if r <= cp_at_mid: # left branch
            upper_bound = mid_index
        else:  # right branch
            lower_bound = mid_index

    cp_at_left = masked_cps[lower_bound]
    result = lower_bound if r <= cp_at_left else upper_bound

    return mask[result]


@nb.jit(nb.int64(nb.float64, nb.float64[:]), nogil=True, nopython=True)
def binary_sample(r, cps):
    """
    Optimized sampler for a Binary Logit Model (e.g. with exactly 2 choices).

    Args:
        r:
        cps:

    Returns:

    """
    return r <= cps[0]


@nb.jit(nb.float64[:](nb.float64[:]), nogil=True, nopython=True)
def multinomial_probabilities(utilities):
    n_cols = len(utilities)
    p = np.zeros(n_cols, dtype=np.float64)  # Return value

    ls = 0.0  # Logsum
    for i, u in enumerate(utilities):
        expu = np.exp(u)
        ls += expu
        p[i] = expu

    for i in range(n_cols):
        p[i] = p[i] / ls

    return p

# region Nested Probabilities


@nb.jit(nb.void(nb.float64[:], _NB_INSTRUCTION_TYPE_1[:]), nopython=True, nogil=True)
def nested_step_1(probabilities, instructions):
    """
    Implements the 'bottom-up' process of computing nested probabilities. Starts with an array of UTILITIES (one cell
    for each alternative) and modifies it in-place to contain conditional probabilities (each sub-nest will sum to 1.0).

    This function is intrinsically tied in the LogitModel._flatten(), which constructs the 'flat' list of instructions
    used in this function.
    """
    logsum = 0.0
    cached_nodes = []
    for record in instructions:
        node_index = record.node_index
        logsum_scale = record.logsum_scale

        if record.logsum_flag:
            # Node index refers to the parent node currently

            # Add the logsum of child nodes to this node's utility
            probabilities[node_index] += logsum_scale * np.log(logsum)

            # Consume the cache of child nodes, converting the exp(u) to conditional probability
            while len(cached_nodes) > 0:
                child_index = cached_nodes.pop()
                probabilities[child_index] /= logsum
            # The cache is now empty

            # Reset the logsum term for the next nest
            logsum = 0.0
        else:
            # Exponentiate a utility, and add it to the on-going logsum
            expu = np.exp(probabilities[node_index]) / logsum_scale
            logsum += expu
            probabilities[node_index] = expu
            cached_nodes.append(node_index)

    # Finalize top-level nodes
    while len(cached_nodes) > 0:
        child_index = cached_nodes.pop()
        probabilities[child_index] /= logsum


@nb.jit(nb.void(nb.float64[:], _NB_INSTRUCTION_TYPE_2[:]), nopython=True, nogil=True)
def nested_step_2(probabilities, instructions):
    """
    Implements the 'top-down' process of multiplying conditional probabilities against one another until the entire
    array sums to 1.0. Parent nodes will get a probability of 0.

    This function is intrinsically tied in the LogitModel._flatten(), which constructs the 'flat' list of instructions
    used in this function.
    """
    # Now we've go conditional probabilities for all choices. But the branch nodes need to be zero'd out can their
    # probabilities applied lower in the nest.
    parent_nodes = set()
    for record in instructions:
        parent_p = probabilities[record.parent_index]
        probabilities[record.child_index] = probabilities[record.child_index] * parent_p
        parent_nodes.add(record.parent_index)
    for pi in parent_nodes:
        probabilities[pi] = 0.0


@nb.jit(nb.float64[:](nb.float64[:], _NB_INSTRUCTION_TYPE_1[:], _NB_INSTRUCTION_TYPE_2[:]), nopython=True, nogil=True)
def nested_probabilities(utilities, instruction_set_1, instruction_set_2):
    """
    Computes probabilities for a nested logit model, from an array of utilities.

    Args:
        utilities:
        instruction_set_1:
        instruction_set_2:

    Returns:

    """
    probabilities = utilities.copy()

    nested_step_1(probabilities, instruction_set_1)
    nested_step_2(probabilities, instruction_set_2)

    return probabilities

# endregion

# region Mid-level functions


@nb.jit(nb.void(nb.float64[:, :], nb.float64[:, :], _NB_INSTRUCTION_TYPE_1[:], _NB_INSTRUCTION_TYPE_2[:], nb.int64[:, :]),
        nogil=True, nopython=True)
def sample_nested_worker(utilities, random_numbers, instruction_set_1, instruction_set_2, out):
    nrows , n_draws = random_numbers.shape

    for i in range(nrows):
        util_row = utilities[i, :]
        probabilities = nested_probabilities(util_row, instruction_set_1, instruction_set_2)
        probabilities = np.cumsum(probabilities)  # Convert to cumulative sum

        for j in range(n_draws):
            r = random_numbers[i, j]
            result = logarithmic_search(r, probabilities)
            out[i, j] = result


@nb.jit(nb.void(nb.float64[:, :], nb.float64[:, :], nb.int64[:, :]), nopython=True, nogil=True)
def sample_multinomial_worker(utilities, random_numbers, out):
    nrows, n_draws = random_numbers.shape

    for i in range(nrows):
        util_row = utilities[i, :]
        probabilities = multinomial_probabilities(util_row)
        probabilities = np.cumsum(probabilities)

        for j in range(n_draws):
            r = random_numbers[i, j]
            result = logarithmic_search(r, probabilities)
            out[i, j] = result


@nb.jit(nb.void(nb.float64[:, :], _NB_INSTRUCTION_TYPE_1[:], _NB_INSTRUCTION_TYPE_2[:], nb.float64[:, :]),
        nopython=True, nogil=True)
def stochastic_nested_worker(utilities, instruction_set_1, instruction_set_2, out):
    nrows = utilities.shape[0]

    for i in range(nrows):
        util_row = utilities[i, :]
        probabilities = nested_probabilities(util_row, instruction_set_1, instruction_set_2)
        out[i, :] = probabilities


@nb.jit(nb.void(nb.float64[:, :], nb.float64[:, :]), nopython=True, nogil=True)
def stochastic_multinomial_worker(utilities, out):
    nrows = utilities.shape[0]

    for i in range(nrows):
        util_row = utilities[i, :]
        probabilities = multinomial_probabilities(util_row)
        out[i, :] = probabilities


# @nb.jit(nb.void(nb.int64[:, :], nb.float64[:], nb.int64[:]), nogil=True, nopython=True)
def weighted_sample_worker(weights, random_numbers, out):
    nrows = weights.shape[0]

    for i in range(nrows):
        row = weights[i, :]
        total = row.sum()
        cps = np.cumsum(row / total)
        r = random_numbers[i]

        index = logarithmic_search(r, cps)
        out[i] = index

# endregion

