import numpy as np
from typing import List
from log_parser.model import IterationMetrics, AggregatedMetrics
from log_parser.parsing import parse_model_info, parse_iterations


def compute_aggregated_metrics(iterations: List[IterationMetrics]) -> AggregatedMetrics:
    """Compute aggregated statistics from a list of IterationMetrics."""
    agg = AggregatedMetrics()
    try:
        if iterations:
            # Create lists from iteration attributes for aggregation
            original_solve_times = [it.original_solve_time for it in iterations]
            permuted_solve_times = [it.permuted_solve_time for it in iterations]
            canonical_original_solve_times = [it.canonical_original_solve_time for it in iterations]
            canonical_permuted_solve_times = [it.canonical_permuted_solve_time for it in iterations]
            original_work_units = [it.original_work_units for it in iterations]
            permuted_work_units = [it.permuted_work_units for it in iterations]
            permutation_distances_before = [it.permutation_distance_before for it in iterations]
            permutation_distances_after = [it.permutation_distance_after for it in iterations]
            models_equivalent = [it.models_equivalent for it in iterations]
            variable_counts_match = [it.variable_counts_match for it in iterations]
            constraint_counts_match = [it.constraint_counts_match for it in iterations]
            
            # Solve Time Metrics
            agg.avg_original_solve_time = np.mean(original_solve_times)
            agg.std_original_solve_time = np.std(original_solve_times)
            agg.avg_permuted_solve_time = np.mean(permuted_solve_times)
            agg.std_permuted_solve_time = np.std(permuted_solve_times)
            agg.avg_canonical_original_solve_time = np.mean(canonical_original_solve_times)
            agg.std_canonical_original_solve_time = np.std(canonical_original_solve_times)
            agg.avg_canonical_permuted_solve_time = np.mean(canonical_permuted_solve_times)
            agg.std_canonical_permuted_solve_time = np.std(canonical_permuted_solve_times)
            
            # Work Units Metrics
            agg.avg_original_work_units = np.mean(original_work_units)
            agg.std_original_work_units = np.std(original_work_units)
            agg.avg_permuted_work_units = np.mean(permuted_work_units)
            agg.std_permuted_work_units = np.std(permuted_work_units)
            
            # Permutation Distance Metrics
            agg.perm_distance_before_avg = np.mean(permutation_distances_before)
            agg.perm_distance_before_std = np.std(permutation_distances_before)
            agg.perm_distance_after_avg = np.mean(permutation_distances_after)
            agg.perm_distance_after_std = np.std(permutation_distances_after)
            
            # Consistency Metrics as percentages
            agg.models_equivalent_pct = np.mean(models_equivalent) * 100
            agg.variable_counts_match_pct = np.mean(variable_counts_match) * 100
            agg.constraint_counts_match_pct = np.mean(constraint_counts_match) * 100
    except Exception as e:
        print(f"Error computing aggregated metrics: {e}")
    
    return agg