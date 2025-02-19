import numpy as np
from typing import List
from log_parser.model import IterationMetrics, AggregatedMetrics
from log_parser.parsing import parse_model_info, parse_iterations


def compute_aggregated_metrics(iterations: List[IterationMetrics]) -> AggregatedMetrics:
    """Compute aggregated statistics from a list of IterationMetrics."""
    agg = AggregatedMetrics()
    try:
        if iterations:
            # --- Solve Times Aggregation ---
            # Use the first iteration's original values (since they're the same in every iteration)
            unique_original_solve_time = iterations[0].original_solve_time
            unique_canonical_original_solve_time = iterations[0].canonical_original_solve_time
            permuted_solve_times = [it.permuted_solve_time for it in iterations]
            canonical_permuted_solve_times = [it.canonical_permuted_solve_time for it in iterations]
            all_permutation_solve_times = [unique_original_solve_time] + permuted_solve_times
            all_canonical_solve_times = [unique_canonical_original_solve_time] + canonical_permuted_solve_times

            # --- Work Units Aggregation ---
            # Use the first iteration's original work units as the unique "original" value.
            unique_original_work_units = iterations[0].original_work_units
            unique_canonical_original_work_units = iterations[0].canonical_original_work_units
            permuted_work_units = [it.permuted_work_units for it in iterations]
            canonical_permuted_work_units = [it.canonical_permuted_work_units for it in iterations]
            all_permutation_work_units = [unique_original_work_units] + permuted_work_units
            all_canonical_work_units = [unique_canonical_original_work_units] + canonical_permuted_work_units

            # Compute the standard deviations
            agg.std_all_permutation_solve_time = np.std(all_permutation_solve_times)
            agg.std_all_canonical_solve_time = np.std(all_canonical_solve_times)

            agg.std_all_permutation_work_units = np.std(all_permutation_work_units)
            agg.std_all_canonical_work_units = np.std(all_canonical_work_units)

            # --- Permut Distance Aggregation ---
            permutation_distances_before = [it.permutation_distance_before for it in iterations]
            permutation_distances_after = [it.permutation_distance_after for it in iterations]

            agg.std_perm_distance_before = np.std(permutation_distances_before)
            agg.std_perm_distance_after = np.std(permutation_distances_after)
            
            # Consistency Metrics as percentages
            models_equivalent = [it.models_equivalent for it in iterations]
            variable_counts_match = [it.variable_counts_match for it in iterations]
            constraint_counts_match = [it.constraint_counts_match for it in iterations]

            agg.models_equivalent_pct = np.mean(models_equivalent) * 100
            agg.variable_counts_match_pct = np.mean(variable_counts_match) * 100
            agg.constraint_counts_match_pct = np.mean(constraint_counts_match) * 100
    except Exception as e:
        print(f"Error computing aggregated metrics: {e}")
    
    return agg