# utils/performance_evaluator.py
import numpy as np
from utils.logging_handler import LoggingHandler

class PerformanceEvaluator:
    def __init__(self):
        self.logger = LoggingHandler().get_logger()

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

    def compute_distance_variability_std(self, results):
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
