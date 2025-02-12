from utils.logging_handler import LoggingHandler

class IterationLogger:
    def __init__(self):
        """
        Initialize with a logger instance.
        """
        self.logger = LoggingHandler().get_logger()

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
        self.logger.info(f"- Permutation Distance Before Canonicalization: {iteration_result['permutation_distance_before_canonicalization']}")
        self.logger.info(f"- Permutation Distance After Canonicalization: {iteration_result['permutation_distance_after_canonicalization']}")

    def log_model_comparison(self, original_canonical, permuted_canonical):
        self.logger.debug("Detailed model comparison:")
        # Assuming your LoggingHandler has a log_model_differences method.
        from utils.logging_handler import LoggingHandler
        LoggingHandler().log_model_differences(self.logger, original_canonical, permuted_canonical)
