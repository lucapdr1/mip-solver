import numpy as np
from typing import List
from log_parser.model import ModelInfo, IterationMetrics, PairwiseDistances, GranularityStats, AggregatedMetrics, AggragatedPairwiseDistances, AggregatedGranularityMetrics
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

            # Compute percentage reduction in std deviation for permutation distance
            agg.std_perm_distance_reduction_pct = round((
                (agg.std_perm_distance_before - agg.std_perm_distance_after) /
                agg.std_perm_distance_before * 100
            ),2)

            # Compute percentage reduction in std deviation for solve times,
            # checking to avoid division by zero if the solver is disabled
            if agg.std_all_permutation_solve_time and agg.std_all_permutation_solve_time != 0:
                agg.std_solve_time_reduction_pct = round((
                    (agg.std_all_permutation_solve_time - agg.std_all_canonical_solve_time) /
                    agg.std_all_permutation_solve_time * 100
                ),2)
            else:
                agg.std_solve_time_reduction_pct = None

            # Compute percentage reduction in std deviation for work units,
            # similarly checking to avoid division by zero if necessary
            if agg.std_all_permutation_work_units and agg.std_all_permutation_work_units != 0:
                agg.std_work_units_reduction_pct = round((
                    (agg.std_all_permutation_work_units - agg.std_all_canonical_work_units) /
                    agg.std_all_permutation_work_units * 100
                ),2)
            else:
                agg.std_work_units_reduction_pct = None

            agg.models_equivalent_pct = np.mean(models_equivalent) * 100
            agg.variable_counts_match_pct = np.mean(variable_counts_match) * 100
            agg.constraint_counts_match_pct = np.mean(constraint_counts_match) * 100
    except Exception as e:
        print(f"Error computing aggregated metrics: {e}")
    
    return agg

def compute_pairwise_metrics(pairwise_stats : PairwiseDistances):
    agg = AggragatedPairwiseDistances()
    try:
        if pairwise_stats.std_pairwise_distance_before:
            agg.std_pairwise_distance_reduction_pct = round((
                    (pairwise_stats.std_pairwise_distance_before - pairwise_stats.std_pairwise_distance_after) /
                    pairwise_stats.std_pairwise_distance_before * 100
                ),2)
        else:
            agg.std_pairwise_distance_reduction_pct = None
        
    except Exception as e:
        print(f"Error computing aggregated metrics: {e}")
    return agg

def compute_granularity_metrics(granularity_stats : GranularityStats, model_info : ModelInfo):
    agg = AggregatedGranularityMetrics()
    try:
        agg.avg_block_percentage = granularity_stats.avg_block_size / (model_info.variables * model_info.constraints) * 100
    except Exception as e:
        print(f"Error computing aggregated metrics: {e}")
    return agg