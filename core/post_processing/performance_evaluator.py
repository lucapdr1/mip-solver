# utils/performance_evaluator.py
from itertools import combinations
import numpy as np
from utils.logging_handler import LoggingHandler

class PerformanceEvaluator:
    def __init__(self):
        self.logger = LoggingHandler().get_logger()

    def compute_work_unit_variability_std(self, results):
        """
        Computes and logs the standard deviation of work units from the experiment results.
        """
        if len(results) < 2:
            self.logger.warning("Not enough runs to compute standard deviation of work units.")
            return

        # Use the original work units from the first run (assumed constant)
        original_work_units = results[0].get('original_work_units', None)
        canonical_from_original_work_units = results[0].get('canonical_from_original_work_units', None)

        if original_work_units is None or canonical_from_original_work_units is None:
            self.logger.warning("Work unit data missing in some results. Skipping analysis.")
            return

        # Collect work units from permuted and canonical solve processes
        permuted_work_units = [r.get('permuted_work_units', None) for r in results] + [original_work_units]
        canonical_from_permuted_work_units = [r.get('canonical_from_permuted_work_units', None) for r in results] + [canonical_from_original_work_units]

        # Remove any None values in case work units were missing in some iterations
        permuted_work_units = [wu for wu in permuted_work_units if wu is not None]
        canonical_from_permuted_work_units = [wu for wu in canonical_from_permuted_work_units if wu is not None]

        if len(permuted_work_units) < 2 or len(canonical_from_permuted_work_units) < 2:
            self.logger.warning("Not enough valid work unit data to compute standard deviation.")
            return

        # Compute standard deviations
        std_permuted = np.std(permuted_work_units, ddof=1)
        std_canon_perm = np.std(canonical_from_permuted_work_units, ddof=1)

        # Log results
        self.logger.info("Work Unit Variability (Standard Deviation):")
        self.logger.info(f" - Std(Original + Permuted Work Units): {std_permuted:.10f}")
        self.logger.info(f" - Std(Canonical-from-Original + Canonical-from-Permuted Work Units): {std_canon_perm:.10f}")

        if std_canon_perm < std_permuted:
            self.logger.info("Canonical form reduces work unit variability across permutations.")
        else:
            self.logger.warning("Canonical form does NOT sufficiently reduce work unit variability across permutations.")

    def compute_solve_time_variability_std(self, results):
        """
        Computes and logs the standard deviation of solve times from the experiment results.
        """
        if len(results) < 2:
            self.logger.warning("Not enough runs to compute standard deviation of solve times.")
            return

        # Use the original solve time from the first run (assumed constant)
        original_solve_time = results[0]['original_solve_time']
        canonical_from_original_solve_time = results[0]['canonical_from_original_solve_time']

        # Collect permuted and canonical solve times
        permuted_solve_times = [r['permuted_solve_time'] for r in results] + [original_solve_time]
        canonical_from_permuted_solve_times = [r['canonical_from_permuted_solve_time'] for r in results] + [canonical_from_original_solve_time]

        std_permuted = np.std(permuted_solve_times, ddof=1)
        std_canon_perm = np.std(canonical_from_permuted_solve_times, ddof=1)

        self.logger.info("Solve-Time Variability (Standard Deviation):")
        self.logger.info(f" - Std(Original + Permuted Solve Times): {std_permuted:.6f}")
        self.logger.info(f" - Std(Canonical-from-Original + Canonical-from-Permuted Solve Times): {std_canon_perm:.6f}")

        if std_canon_perm < std_permuted:
            self.logger.info("Canonical form reduces solve-time variability across permutations.")
        else:
            self.logger.warning("Canonical form does NOT sufficiently reduce solve-time variability across permutations.")

    def compute_simple_distance_variability_std(self, results):
        """
        Computes and logs the standard deviation of permutation distances.
        """
        if len(results) < 2:
            self.logger.warning("Not enough runs to compute standard deviation of permutation distances.")
            return

        distances_before = [r['permutation_distance_before_canonicalization'] for r in results]
        distances_after = [r['permutation_distance_after_canonicalization'] for r in results]

        std_before = np.std(distances_before, ddof=1)
        std_after = np.std(distances_after, ddof=1)

        self.logger.info("Permutation Distance Variability (Standard Deviation):")
        self.logger.info(f" - Std(Permutation Distance Before Canonicalization): {std_before:.6f}")
        self.logger.info(f" - Std(Permutation Distance After Canonicalization): {std_after:.6f}")

        if std_after < std_before:
            self.logger.info("Canonicalization reduces permutation distance variability across permutations.")
        else:
            self.logger.warning("Canonicalization does NOT sufficiently reduce permutation distance variability.")
    
    def compute_all_pairs_distance_variability_std(self, permut_storage):
        """
        Computes and logs the standard deviation of permutation distances across all pairwise comparisons.
        Assumes that `permut_storage` has two lists:
        - permut_storage.permutations: list of (var_order, constr_order) tuples before canonicalization
        - permut_storage.canonical_forms: list of (var_order, constr_order) tuples after canonicalization
        """
        n = len(permut_storage)
        if n <= 2:
            self.logger.warning("Not enough runs to compute standard deviation of permutation distances (all pairs).")
            return

        distances_before = []
        distances_after = []

        # Iterate over all unique pairs using indices 0 through n-1
        for i, j in combinations(range(n), 2):
            p_constr1, p_var1 = permut_storage.get_permutation(i)
            p_constr2, p_var2 = permut_storage.get_permutation(j)
            c_constr1, c_var1 = permut_storage.get_canonical_form(i)
            c_constr2, c_var2 = permut_storage.get_canonical_form(j)

            # Compute the distances for the pair (assumes you have defined self.compute_permutation_distance)
            distance_before = permut_storage.compute_permutation_distance(p_constr1, p_var1, p_constr2, p_var2)
            distance_after = permut_storage.compute_permutation_distance(c_constr1, c_var1, c_constr2, c_var2)

            distances_before.append(distance_before)
            distances_after.append(distance_after)

        std_before = np.std(distances_before, ddof=1)
        std_after = np.std(distances_after, ddof=1)

        self.logger.info("All-Pairs Permutation Distance Variability (Standard Deviation):")
        self.logger.info(f" - Std(All-Pairs Distance Before Canonicalization): {std_before:.6f}")
        self.logger.info(f" - Std(All-Pairs Distance After Canonicalization): {std_after:.6f}")

        if std_after < std_before:
            self.logger.info("Canonicalization reduces permutation distance variability across all pairwise comparisons.")
        else:
            self.logger.warning("Canonicalization does NOT sufficiently reduce permutation distance variability across all pairwise comparisons.")

