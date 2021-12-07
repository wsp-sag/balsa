from .general import align_categories, is_identifier, reindex_series, sum_df_sequence, sort_nicely
from .io import *
from .matrices import (fast_unstack, fast_stack, aggregate_matrix, matrix_balancing_1d, matrix_balancing_2d,
                       matrix_bucket_rounding, split_zone_in_matrix, disaggregate_matrix)
from .modelling import tlfd, distance_array, distance_matrix

try:
    from .plotting import trumpet_diagram, convergence_boxplot, location_summary
except ImportError:
    pass
