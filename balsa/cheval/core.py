from __future__ import division, absolute_import, print_function
# Note: unicode literals is deliberately NOT imported because it causes an error with np.dtype in Py2

import numba as nb
import numpy as np


MIN_RANDOM_VALUE = np.finfo(np.float64).tiny


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

    # The check below is required to avoid a very specific edge case in which there is more than one 0-probability
    # choice at the start of the probability array, e.g. [0, 0, 0, 0.1, 0.3, 0.7, 1.0]. The randomizer draws on the
    # interval [0, 1), so it's a (very) small possibility, but nonetheless would yield potentially very wrong results
    if r == 0:
        r = MIN_RANDOM_VALUE

    ncols = len(cps)

    lower_bound, upper_bound = 0, ncols - 1
    while (upper_bound - lower_bound) > 1:
        mid_index = np.uint32((upper_bound + lower_bound) // 2)
        cp_at_mid = cps[mid_index]
        if r <= cp_at_mid:  # left branch
            upper_bound = mid_index
        else:  # right branch
            lower_bound = mid_index

    cp_at_left = cps[lower_bound]
    if r <= cp_at_left:
        return lower_bound
    else:
        return upper_bound


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


@nb.jit(nb.void(nb.float64[:]), nogil=True, nopython=True)
def cumsum(array):
    accum = 0.0
    length = len(array)
    for i in range(length):
        accum += array[i]
        array[i] = accum

# region Nested Probabilities


@nb.jit(nb.float64[:](nb.float64[:], nb.int64[:], nb.int64[:], nb.float64[:]), nopython=True, nogil=True)
def nested_probabilities(utilities, hierarchy, levels, logsum_scales):
    n_cells = len(utilities)
    probabilities = utilities.copy()
    top_logsum = 0
    logsums = np.zeros(n_cells, dtype=np.float64)

    # Step 1: Exponentiate the utilities and collect logsums
    max_level =levels.max()
    current_level = max_level
    for _ in range(max_level + 1):
        # Go through levels in reverse order (e.g. starting at the bottom)
        for index, level in enumerate(levels):
            if level != current_level: continue
            parent = hierarchy[index]

            existing_logsum = logsums[index]
            parent_ls_scale = logsum_scales[parent] if parent >= 0 else 1.0
            if existing_logsum != 0:
                current_ls_scale = logsum_scales[index]
                expu = np.exp((probabilities[index] + current_ls_scale * np.log(existing_logsum)) / parent_ls_scale)
            else:
                expu = np.exp(probabilities[index] / parent_ls_scale)
            if parent >= 0: logsums[parent] += expu
            else: top_logsum += expu
            probabilities[index] = expu
        current_level -= 1

    # Step 2: Use logsums to compute conditional probabilities
    for index, parent in enumerate(hierarchy):
        ls = top_logsum if parent == -1 else logsums[parent]
        probabilities[index] = probabilities[index] / ls

    # Step 3: Compute absolute probabilities for child nodes, collecting parent nodes
    for current_level in range(1, max_level + 1):
        for index, level in enumerate(levels):
            if level != current_level: continue
            parent = hierarchy[index]
            probabilities[index] *= probabilities[parent]

    # Step 4: Zero-out parent node probabilities
    # This does not use a Set because Numba sets are really slow
    for parent in hierarchy:
        if parent < 0: continue
        probabilities[parent] = 0.0

    return probabilities


# endregion

# region Mid-level functions


@nb.jit(nb.void(nb.float64[:], nb.float64[:], nb.int64[:], nb.int64[:], nb.float64[:], nb.int64[:]), nopython=True,
        nogil=True)
def sample_nested(utility_row, random_numbers, hierarchy, levels, logsum_scales, out):
    probabilities = nested_probabilities(utility_row, hierarchy, levels, logsum_scales)
    cumsum(probabilities)  # Convert to cumulative sum

    for i, r in enumerate(random_numbers):
        result = logarithmic_search(r, probabilities)
        out[i] = result


@nb.jit(nb.void(nb.float64[:], nb.float64[:], nb.int64[:]), nopython=True, nogil=True)
def sample_multinomial(utility_row, random_numbers, out):
    probabilities = multinomial_probabilities(utility_row)
    cumsum(probabilities)

    for i, r in enumerate(random_numbers):
        result = logarithmic_search(r, probabilities)
        out[i] = result


@nb.jit(nb.void(nb.float64[:], nb.float64[:], nb.int64[:]), nopython=True, nogil=True)
def weighted_sample(weights, random_numbers, out):
    total = weights.sum()
    cps = weights / total
    cumsum(cps)

    for i, r in enumerate(random_numbers):
        result = logarithmic_search(r, cps)
        out[i] = result


# endregion

# region High-level functions (to be wrapped in Threads)


@nb.jit(nb.void(nb.float64[:, :], nb.float64[:, :], nb.int64[:], nb.int64[:], nb.float64[:], nb.int64[:, :]),
        nogil=True, nopython=True)
def sample_nested_worker(utilities, random_numbers, hierarchy, levels, logsum_scales, out):
    nrows, n_draws = random_numbers.shape

    for i in range(nrows):
        util_row = utilities[i, :]
        sample_nested(util_row, random_numbers[i, :], hierarchy, levels, logsum_scales, out[i, :])


@nb.jit(nb.void(nb.float64[:, :], nb.float64[:, :], nb.int64[:, :]), nopython=True, nogil=True)
def sample_multinomial_worker(utilities, random_numbers, out):
    nrows, n_draws = random_numbers.shape

    for i in range(nrows):
        util_row = utilities[i, :]
        sample_multinomial(util_row, random_numbers[i, :], out[i, :])


@nb.jit(nb.void(nb.float64[:, :],  nb.int64[:], nb.int64[:], nb.float64[:], nb.float64[:, :]),
        nopython=True, nogil=True)
def stochastic_nested_worker(utilities, hierarchy, levels, logsum_scales, out):
    nrows = utilities.shape[0]

    for i in range(nrows):
        util_row = utilities[i, :]
        probabilities = nested_probabilities(util_row, hierarchy, levels, logsum_scales)
        out[i, :] = probabilities


@nb.jit(nb.void(nb.float64[:, :], nb.float64[:, :]), nopython=True, nogil=True)
def stochastic_multinomial_worker(utilities, out):
    nrows = utilities.shape[0]

    for i in range(nrows):
        util_row = utilities[i, :]
        probabilities = multinomial_probabilities(util_row)
        out[i, :] = probabilities


@nb.jit(nb.void(nb.float64[:, :], nb.float64[:, :], nb.int64[:, :]), nogil=True, nopython=True)
def weighted_sample_worker(weights, random_numbers, out):
    nrows = weights.shape[0]

    for i in range(nrows):
        weighted_sample(weights[i, :], random_numbers[i, :], out[i, :])

# endregion

