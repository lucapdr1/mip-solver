from utils.logging_handler import LoggingHandler
from utils.config import LOG_LEVEL, INPUT_DIR, OUTPUT_DIR, INPUT_PROBLEM, PERMUTE_ORIGINAL, NUMBER_OF_PERMUTATIONS, NORMALIZATION_ACTIVE, MATRICES_TO_CSV, LOG_MATRIX, LOG_MODEL_COMPARISON, PRODUCTION, RECURSIVE_RULES, DISABLE_SOLVING, MAX_SOLVE_TIME

class IterationLogger:
    def __init__(self):
        """
        Initialize with a logger instance.
        """
        self.logger = LoggingHandler().get_logger()

    def log_experiment_settings(self):
        self.logger.info("Experiment Settings:")
        self.logger.info(f"- INPUT_DIR: {INPUT_DIR}")
        self.logger.info(f"- LOG_LEVEL: {LOG_LEVEL}")
        self.logger.info(f"- OUTPUT_DIR: {OUTPUT_DIR}")
        self.logger.info(f"- INPUT_PROBLEM: {INPUT_PROBLEM}")
        self.logger.info(f"- PERMUTE_ORIGINAL: {PERMUTE_ORIGINAL}")
        self.logger.info(f"- NUMBER_OF_PERMUTATIONS: {NUMBER_OF_PERMUTATIONS}")
        self.logger.info(f"- NUMBER_OF_PERMUTATIONS: {NUMBER_OF_PERMUTATIONS}")
        self.logger.info(f"- NORMALIZATION_ACTIVE: {NORMALIZATION_ACTIVE}")
        self.logger.info(f"- MATRICES_TO_CSV: {MATRICES_TO_CSV}")
        self.logger.info(f"- LOG_MATRIX: {LOG_MATRIX}")
        self.logger.info(f"- LOG_MODEL_COMPARISON: {LOG_MODEL_COMPARISON}")
        self.logger.info(f"- PRODUCTION: {PRODUCTION}")
        self.logger.info(f"- RECURSIVE_RULES: {RECURSIVE_RULES}")
        self.logger.info(f"- DISABLE_SOLVING: {DISABLE_SOLVING}")
        self.logger.info(f"- MAX_SOLVE_TIME: {MAX_SOLVE_TIME}")

    def log_model_info(self, model, file_path):
        self.logger.info(f"Successfully loaded problem from {file_path}")
        self.logger.info("Problem Details:")
        self.logger.info(f"- Variables: {model.NumVars}")
        self.logger.info(f"- Constraints: {model.NumConstrs}")
        sense = "Minimize" if model.ModelSense == 1 else "Maximize"
        self.logger.info(f"- Objective Sense: {sense}")

    def log_iteration_results(self, iteration_num, iteration_result):
        self.logger.info(f"Iteration {iteration_num} Results:")
        self.logger.info(f"- Models equivalent: {iteration_result['equivalent']}")
        self.logger.info(f"- Variable counts match: {iteration_result['original_vars'] == iteration_result['permuted_vars']}")
        self.logger.info(f"- Constraint counts match: {iteration_result['original_constrs'] == iteration_result['permuted_constrs']}")
        self.logger.info(f"- Original Objective Value: {iteration_result['original_objective']}")
        self.logger.info(f"- Permuted Objective Value: {iteration_result['permuted_objective']}")
        self.logger.info(f"- Canonical from Original Objective Value: {iteration_result['canonical_from_original_objective']}")
        self.logger.info(f"- Canonical from Permuted Objective Value: {iteration_result['canonical_from_permuted_objective']}")
        self.logger.info(f"- Original Solve Time: {iteration_result['original_solve_time']:.10f} seconds")
        self.logger.info(f"- Permuted Solve Time: {iteration_result['permuted_solve_time']:.10f} seconds")
        self.logger.info(f"- Canonical from Original Solve Time: {iteration_result['canonical_from_original_solve_time']:.10f} seconds")
        self.logger.info(f"- Canonical from Permuted Solve Time: {iteration_result['canonical_from_permuted_solve_time']:.10f} seconds")
        self.logger.info(f"- Original Work Units: {iteration_result['original_work_units']:.10f}")
        self.logger.info(f"- Permuted Work Units: {iteration_result['permuted_work_units']:.10f}")
        self.logger.info(f"- Canonical from Original Work Units: {iteration_result['canonical_from_original_work_units']:.10f}")
        self.logger.info(f"- Canonical from Permuted Work Units: {iteration_result['canonical_from_permuted_work_units']:.10f}")
        self.logger.info(f"- Permutation Distance Before Canonicalization: {iteration_result['permutation_distance_before_canonicalization']}")
        self.logger.info(f"- Permutation Distance After Canonicalization: {iteration_result['permutation_distance_after_canonicalization']}")

    def log_model_comparison(self, original_canonical, permuted_canonical):
        self.logger.lazy_debug("Detailed model comparison:")
        # Assuming your LoggingHandler has a log_model_differences method.
        from utils.logging_handler import LoggingHandler
        LoggingHandler().log_model_differences(self.logger, original_canonical, permuted_canonical)

    def log_granularity_stats(self, stats):
        """Logs granularity statistics for block sizes."""
        if stats:
            self.logger.info("Granularity Statistics:")
            self.logger.info(" - Total Blocks: %d", stats['block_sizes']['total_blocks'])
            self.logger.info(" - Avg(Block Size): %.4f", stats['block_sizes']['average'])
            self.logger.info(" - Min(Block Size): %d", stats['block_sizes']['min'])
            self.logger.info(" - Max(Block Size): %d", stats['block_sizes']['max'])
            self.logger.info(" - Sum of SubBlocks sizes: %d", stats['block_sizes']['sum_of_block_sizes'])
            self.logger.info(" - Original matrix size: %d", stats['block_sizes']['original_matrix_size'])
        else:
            self.logger.warning("No granularity data collected.")