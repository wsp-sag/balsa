from multiprocessing import cpu_count
from threading import Thread
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy import float32 as nfloat
from numpy import inf
from numpy import int16 as nshort
from numpy import ndarray
from pandas import DataFrame, Index, MultiIndex

try:
    from numba import njit
    NUMBA_LOADED = True
except ImportError:
    njit = None
    NUMBA_LOADED = False


_NULL_INDEX = -1
_NEG_INF = -inf


if NUMBA_LOADED:
    def _get_breaks(n_items: int, n_workers: int) -> List[Tuple[int, int]]:
        """Gets starts and stops for breaks similar to numpy.array_slice()"""
        div, remainder = divmod(n_items, n_workers)
        slices = []
        bottom = 0
        for i in range(n_workers):
            top = bottom + div
            top += int(i < remainder)
            slices.append((bottom, top))
            bottom = top
        return slices


    @njit
    def _update_heap(utilities: ndarray, zones: ndarray, new_u: float, new_zone: int):
        """Inserts a value into a sorted array, maintaining sort order, and the number of items in that array. The array
        is sorted lowest-to-highest"""
        i = 0
        top = len(utilities)
        while i < top:
            current_u = utilities[i]
            if new_u < current_u:
                break
            i += 1
        if i <= 0:
            return
        for j in range(i - 1):
            utilities[j] = utilities[j + 1]
            zones[j] = zones[j + 1]
        utilities[i - 1] = new_u
        zones[i - 1] = new_zone


    @njit(nogil=True)
    def _nbf_twopart_worker(access_utils: ndarray, egress_utils: ndarray, result_utils: ndarray,
                            result_stations: ndarray, start: int, stop: int, k: int):
        """Performance-tuned NBF to operate on its own thread"""
        n_origins, n_intermediate = access_utils.shape
        n_destinations = egress_utils.shape[1]

        # Allocate the sorted heap of utilities (and associated indices) once per thread
        util_heap = np.full(shape=k, fill_value=-inf)
        zones_heap = np.full(shape=k, fill_value=_NULL_INDEX)

        for offset in range(start, stop):
            origin_zone, destination_zone = divmod(offset, n_destinations)

            # Reset the heap for this OD
            util_heap[:] = _NEG_INF
            zones_heap[:] = _NULL_INDEX

            for interim_zone in range(n_intermediate):

                if interim_zone >= n_intermediate:
                    print("ERR", offset, origin_zone, destination_zone, interim_zone)
                    raise AssertionError()

                interim_util = access_utils[origin_zone, interim_zone] + egress_utils[interim_zone, destination_zone]

                # In general, for problems where (n_origins * n_destinations) >> k, most values will not be in the top
                # k. So quickly check against the lowest utility in the heap to avoid calling the updater func
                if interim_util < util_heap[0] or interim_util == _NEG_INF:
                    continue
                _update_heap(util_heap, zones_heap, interim_util, interim_zone)

            result_utils[origin_zone, destination_zone, :] = util_heap
            result_stations[origin_zone, destination_zone, :] = zones_heap


    def _validate_access_egress_tables(access_table: DataFrame, egress_table: DataFrame) -> Tuple[Index, Index, Index]:

        assert access_table.index.nlevels == 2, "Access table index must have two levels"
        assert egress_table.index.nlevels == 2, "Egress table index must have two levels"

        # Take the unique index of each level, as Pandas can return more items than is present, if the frame is a slice
        origin_zones: Index = access_table.index.unique(level=0)
        intermediate_zones: Index = access_table.index.unique(level=1)
        destination_zones: Index = egress_table.index.unique(level=1)

        # Check that the access and egress tables have compatible indices
        assert intermediate_zones.equals(egress_table.index.unique(level=0)), \
            "Access index level 2 and egress index level 1 must be the same"

        return origin_zones, intermediate_zones, destination_zones


    def best_intermediate_zones(access_table: DataFrame, egress_table: DataFrame, cost_column: str, k: int = 1,
                                n_threads: int = None, squeeze=True, other_columns=True,
                                intermediate_name: str = "intermediate_zone", maximize=True,
                                availability_column: str = "available", null_index=0
                                ) -> Union[DataFrame, Dict[int, DataFrame]]:
        """Numba-accelerated.

        Triple-index operation for two matrices, finding the most- or least-cost intermediate zones. Takes an access
        matrix of the shape (O, I) and an egress matrix of the shape (I, D) to produce a combined matrix of the shape
        (O, D), with the best intermediate I. Also works to construct multiple (O, D) matrices - for the top _k_
        intermediate zones in _I_.

        There is no restriction on the label dtypes, as long as the access and egress tables share the same _I_ index.

        Both the input matrices must be provided in "tall" format - as Pandas Series with a 2-level MultiIndex.
        Essentially, the access and egress tables are DataFrames with multiple matrices defined within. The output
        table(s) are also returned in a tall format.

        When constructing the result tables, columns in the access and egress tables are "carried forward" such that the
        results columns will be the union of columns in the input tables. Columns in one table only will be carried
        forward unmodified and retain their data type. Columns in both tables will be added together, and thus MUST be
        numeric.

        In the specified cost column, a value of `-inf` (or `inf` when minimizing) is respected as the sentinel value
        for unavailable. (O, I) or (I, D) interchanges with this sentinel value will not be considered.

        Args:
            access_table: DataFrame with 2-level MultiIndex of the shape ((O, I), A). Must include the specified cost
                column
            egress_table: DataFrame with 2-level MultiIndex of the shape ((I, D), E). Must include the specified cost
                column
            cost_column: Name of the column in the access and egress table to use as the cost to minimize/maximize.
                Values of `+/- inf` are respected to indicate unavailable choices
            k: The number of ranks to return (e.g., find the _k_ best intermediate zones). If k <= 0, it will be
                corrected to 1.
            n_threads: Number of threads to use. Defaults to cpu_count()
            squeeze: If k == 1 and squeeze=True, a single DataFrame is returned. Otherwise, a Dictionary of DataFrames
                will be returned.
            other_columns: If True, the result DataFrame will include all columns in the access and egress tables. The
                result table will be of the shape ((O, D), A | E + 3)
            intermediate_name: Name of the column in the result table containing the selected intermediate zone.
            maximize: If True, this function maximize the result. If False, it minimizes it.
            availability_column: Name of the column in the result table containing a flag whether ANY intermediate zone
                was found to be available.
            null_index: Fill value used if NO intermediate zone is available.

        Returns:
            DataFrame: If k == 1 and squeeze=True. A DataFrame of the shape ((O, D), A | E + 3), containing the
                intermediate zone selected, the associated max/min cost, and a flag indicating its availability.
                Additional columns from the access and egress tables, indexed for the appropriately chosen intermediate
                zone, will also be included if other_columns=True.
            Dict[int, DataFrame]: If k > 1. The keys represent the ranks, so result[1] is the best intermediate zone,
                result[2] is the second-best, etc. The value DataFrames are in the same format as if k == 1, just with
                different intermediate zones chosen.
        """

        # Check inputs
        k = max(1, k)
        if n_threads is None:
            n_threads = cpu_count()
        origins, intermediates, destinations = _validate_access_egress_tables(access_table, egress_table)
        n_origins, n_intermediate, n_destinations = len(origins), len(intermediates), len(destinations)

        # Compute best path(s)
        access_cost = access_table[cost_column].values.reshape([n_origins, n_intermediate]).astype(nfloat)
        egress_cost = egress_table[cost_column].values.reshape([n_intermediate, n_destinations]).astype(nfloat)

        if not maximize:  # Invert the cost to use code set to maximize
            access_cost *= -1
            egress_cost *= -1

        result_cost = np.zeros(shape=(n_origins, n_destinations, k), dtype=nfloat)
        result_indices = np.zeros(dtype=nshort, shape=(n_origins, n_destinations, k))

        # Setup the workers (1 per thread) to select best paths for a subset of ODs
        breaks = _get_breaks(n_origins * n_destinations, n_threads)
        threads = [
            Thread(target=_nbf_twopart_worker, args=[
                access_cost, egress_cost, result_cost, result_indices,
                start, stop, k
            ])
            for start, stop in breaks
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Construct composite result tables
        if other_columns:
            remaining_columns = set(access_table.columns | egress_table.columns) - {cost_column, intermediate_name,
                                                                                    availability_column}
        else:
            remaining_columns = set()

        row_index = MultiIndex.from_product([origins, destinations])        # Labels for the rows
        access_indexer = np.repeat(np.arange(n_origins), n_destinations)    # Indexer for the access table
        egress_indexer = np.tile(np.arange(n_destinations), n_origins)      # Indexer for the egress table
        tables = {}                                                         # Results

        for i in range(k):
            table = DataFrame(index=row_index)

            offsets_i = result_indices[:, :, i]  # 2D indexer for this (i ∈ k) path
            flat_offsets_i = offsets_i.flatten()  # Convert to 1D indexer
            availability_i = flat_offsets_i != _NULL_INDEX

            intermediate_result_i = intermediates.take(flat_offsets_i)
            intermediate_result_i[~availability_i] = null_index
            table[intermediate_name] = intermediate_result_i

            table[availability_column] = availability_i
            table[cost_column] = result_cost[offsets_i]

            # If there are any columns left, add them to the composite table
            for column in remaining_columns:
                in_access = column in access_table
                in_egress = column in egress_table

                if in_access:
                    access_matrix = access_table[column].values.reshape([n_origins, n_intermediate])
                    access_component = access_matrix[access_indexer, flat_offsets_i]
                if in_egress:
                    egress_matrix = egress_table[column].values.reshape([n_intermediate, n_destinations])
                    egress_component = egress_matrix[flat_offsets_i, egress_indexer]

                if in_access and in_egress:
                    composite_data = access_component + egress_component
                elif in_access:
                    composite_data = access_component
                elif in_egress:
                    composite_data = egress_component
                else:
                    raise RuntimeError("This shouldn't happen")

                table[column] = composite_data

            tables[k - i] = table

        if k == 1 and squeeze:
            return tables[1]
        return tables
else:
    def best_intermediate_zones(*args, **kwargs):
        raise ModuleNotFoundError('Please install Numba to run this function')
