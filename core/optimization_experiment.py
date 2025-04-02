# core/optimization_experiment.py

import os
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import tempfile

from utils.logging_handler import LoggingHandler
from core.problem_transform.problem_permutator import ProblemPermutator, PermutationStorage
from core.canonical_form_generator import CanonicalFormGenerator
from utils.iteration_logger import IterationLogger
from core.post_processing.performance_evaluator import PerformanceEvaluator
from core.problem_transform.problem_scaler import ProblemScaler
from core.problem_transform.problem_normalizer import ProblemNormalizer
from core.problem_transform.distance import KendallTauDistance, WeightedKendallTauDistance, AdjacencyAwareDistance, CompositeDistance
from utils.problem_printer import ProblemPrinter
from utils.plots_handler import save_all_plots
from utils.config import PERMUTE_ORIGINAL, PERMUTE_SEED, PERMUTE_GRANULARITY_K, LOG_MODEL_COMPARISON, LOG_MATRIX, PRODUCTION, BUCKET_NAME, SCALING_ACTIVE, NORMALIZATION_ACTIVE, DISABLE_SOLVING, RECURSIVE_RULES, MAX_SOLVE_TIME, NUMBER_OF_THREADS

class OptimizationExperiment:
    def __init__(self, gp_env, file_path, ordering_rule):
        self.gp_env = gp_env
        self.file_path = file_path
        self.logger = LoggingHandler().get_logger()
    
        self.original_model = self.load_problem()
        self.ordering_rule = ordering_rule

        IterationLogger().log_experiment_settings()
        IterationLogger().log_model_info(self.original_model, file_path)

        self.permutator = ProblemPermutator(gp_env, self.original_model)
        self.canonical_generator = CanonicalFormGenerator(gp_env, self.original_model, self.ordering_rule)

        self.permut_storage = None
         # Lists to store figures
        self.permuted_matrices = []
        self.canonical_matrices = []

        self.row_distance_metric = None
        self.col_distance_metric = None
    
    def run_experiment(self, num_iterations):
        """Run multiple iterations with detailed logging and solving functionality.
        If include_initial is False, the baseline used in comparisons will be generated
        as a permuted problem rather than using the original problem.
        """
        results = []

        if PERMUTE_ORIGINAL:
            self.logger.info("Calculating baseline from a permuted problem (skipping original).")
            #Change parameters to use percentage instead of number of variables per block
            baseline_model, baseline_var_order, baseline_constr_order, _, _ = self.permutator.create_permuted_problem(PERMUTE_GRANULARITY_K, PERMUTE_SEED)
        else:
            self.logger.info("Solving Original Problem")
            baseline_model = self.original_model
            baseline_var_order = np.arange(self.original_model.NumVars)
            baseline_constr_order = np.arange(self.original_model.NumConstrs)
            
        if LOG_MATRIX:
            self.permuted_matrices.append(baseline_model.getA())

        row_adjacency = self.permutator.get_constraint_adjacency(baseline_model)
        cols_adjacency = self.permutator.get_variable_adjacency(baseline_model)
        A_csr = baseline_model.getA().tocsr()
        rcm_adjacency = self.permutator.get_rcm_adjacency(A_csr)
        cluster_assignments = self.permutator.get_cluster_assignments(A_csr)

        #self.row_distance_metric = CompositeDistance(cluster_assignments, rcm_adjacency, alpha_cluster=1.0, beta_local=1.0)
        self.row_distance_metric = KendallTauDistance()
        self.col_distance_metric = KendallTauDistance()

        self.permut_storage = PermutationStorage(self.permutator, self.row_distance_metric, self.col_distance_metric)
        self.permut_storage.add_permutation(baseline_constr_order, baseline_var_order)

        # Solve the baseline problem (either original or permuted)
        baseline_result = self.solve_problem(baseline_model)

        # Generate the canonical form of the baseline model
        self.logger.lazy_debug("Generating canonical form for the baseline model...")
        canon_gen = CanonicalFormGenerator(self.gp_env, baseline_model, self.ordering_rule)
        baseline_canonical, baseline_canonical_var_order, baseline_canonical_constr_order = \
            canon_gen.get_canonical_form()
        
        final_canonical_var_order = baseline_var_order[baseline_canonical_var_order]
        final_canonical_constr_order = baseline_constr_order[baseline_canonical_constr_order]

        self.permut_storage.add_canonical_form(final_canonical_constr_order, final_canonical_var_order)

        ordered_baseline_model = self.permutator.apply_permutation(
            baseline_model, baseline_canonical_var_order, baseline_canonical_constr_order
        )
        if LOG_MATRIX:
            self.canonical_matrices.append(ordered_baseline_model.getA())

        self.logger.info("Solving Canonical from Baseline Problem")
        baseline_canonical_result = self.solve_problem(ordered_baseline_model)

        # Use baseline values in subsequent iterations (named as "original" in later code)
        for i in range(num_iterations):
            self.logger.info(f"Running iteration {i+1}/{num_iterations}")
            try:
                (permuted_canonical, permuted_result, final_ordered_result,
                permuted_distance, canonical_distance) = self.run_single_iteration(
                    baseline_var_order, baseline_constr_order, final_canonical_var_order, final_canonical_constr_order, (i+1)
                )
                iteration_result = self.build_iteration_result(
                    baseline_canonical, permuted_canonical,
                    baseline_result, permuted_result,
                    baseline_canonical_result, final_ordered_result,
                    permuted_distance, canonical_distance
                )
                results.append(iteration_result)
                IterationLogger().log_iteration_results(i + 1, iteration_result)
            except Exception as e:
                self.logger.error(f"Error in iteration {i+1}: {str(e)}")
                raise

        # Compute and log variability metrics
        PerformanceEvaluator().compute_solve_time_variability_std(results)
        PerformanceEvaluator().compute_work_unit_variability_std(results)
        PerformanceEvaluator().compute_simple_distance_variability_std(results)
        PerformanceEvaluator().compute_all_pairs_distance_variability_std(self.permut_storage)

        if RECURSIVE_RULES:
            stats = self.ordering_rule.get_granularity_statistics()
            IterationLogger().log_granularity_stats(stats)
        if LOG_MATRIX:
            save_all_plots(self.permuted_matrices, self.canonical_matrices, self.file_path, "experiment_plots.png")
        return results

    def run_single_iteration(self, baseline_var_order, baseline_constr_order, original_canonical_var_order, original_canonical_constr_order, index):
        try:
            self.logger.lazy_debug("Starting new iteration...")

            # === 1. Create the permuted problem (unscaled) ===
            self.logger.info("Creating Permuted Problem")
            permuted_model, var_permutation, constr_permutation, _, _ = self.permutator.create_permuted_problem(PERMUTE_GRANULARITY_K, (PERMUTE_SEED+index))
            #ProblemPrinterlog_model(permuted_model, self.logger, level="DEBUG")
            if LOG_MATRIX:
                self.permuted_matrices.append(permuted_model.getA())

            self.permut_storage.add_permutation(constr_permutation, var_permutation)

            # Compute permutation distance BEFORE canonicalization (using unscaled permuted model)
            self.logger.info("Computing Permutation Distance before Canonicalization...")

            self.logger.lazy_debug(f"Original Constraint Order: {baseline_constr_order}")
            self.logger.lazy_debug(f"Permuted Constraint Order: {constr_permutation}")
            self.logger.lazy_debug(f"Original Variable Order: {baseline_var_order}")
            self.logger.lazy_debug(f"Permuted Variable Order: {var_permutation}")

            permuted_distance = self.permutator.permutation_distance(
                baseline_constr_order, baseline_var_order,
                constr_permutation, var_permutation,
                self.row_distance_metric,
                self.col_distance_metric,
                alpha=1.0, 
                beta=1.0
            )
            self.logger.info(f"Permutation Distance Before Canonicalization: {permuted_distance}")

            # Solve the unscaled permuted problem (for metric purposes)
            self.logger.info("Solving Permuted Problem")
            permuted_result = self.solve_problem(permuted_model)

            # === 2. Apply scaling if requested ===
            if SCALING_ACTIVE:
                self.logger.info("Scaling is enabled. Scaling the Permuted Problem.")
                scaler = ProblemScaler(self.gp_env, permuted_model)
           
                scaled_model, used_row_scales, used_col_scales, D_row, D_col = scaler.create_scaled_problem_random(
                    n_rows_to_scale=2,
                    n_cols_to_scale=2,
                    row_scale_bounds=(1.2, 1.9),
                    col_scale_bounds=(1.2, 1.9),
                    row_allow_negative=False,
                    col_allow_negative=False
                )
                
                """
                 # Example: Only scale the first constraint by a factor of 2.
                row_scales = [1, 1]       # Only the first constraint is scaled by 2.
                col_scales = [2, 1]    # All columns remain unchanged.

                # Assuming 'scaler' is an instance of ProblemScaler:
                scaled_model, used_row_scales, used_col_scales, D_row, D_col = scaler.create_scaled_problem_with_scales(row_scales, col_scales)
                """
                #ProblemPrinterlog_model(scaled_model, self.logger, level="DEBUG")
                intermediate_model = scaled_model
            else:
                self.logger.info("Scaling is disabled. Using unscaled permuted model.")
                intermediate_model = permuted_model
                used_row_scales = None
                used_col_scales = None

            # === 3. Apply normalization if requested ===
            if NORMALIZATION_ACTIVE:
                self.logger.info("Normalization is enabled. Normalizing the problem.")
                normalizer = ProblemNormalizer()
                normalized_model = normalizer.normalize_model(intermediate_model)
                #ProblemPrinterlog_model(normalized_model, self.logger, level="DEBUG")
            else:
                self.logger.info("Normalization is disabled. Using the intermediate model as is.")
                normalized_model = intermediate_model

            # === 4. Generate canonical form from the final model ===
            self.logger.lazy_debug("Generating canonical form for the final model...")
            # Note: You may want to pass a Normalizer() instance to CanonicalFormGenerator
            # if canonicalization itself makes use of normalization.
            canon_gen = CanonicalFormGenerator(self.gp_env, normalized_model, self.ordering_rule)
            permuted_canonical, permuted_canonical_var_order, permuted_canonical_constr_order = canon_gen.get_canonical_form()


            # === 5. Compose final ordering ===
            # Final ordering is the composition of the original permutation with the canonical ordering.
            final_var_order = var_permutation[permuted_canonical_var_order]
            final_constr_order = constr_permutation[permuted_canonical_constr_order]

            self.permut_storage.add_canonical_form(final_constr_order, final_var_order)

            self.logger.info("Computing Permutation Distance after Canonicalization...")
            self.logger.lazy_debug(f"Original Canonical Constraint Order: {original_canonical_constr_order}")
            self.logger.lazy_debug(f"Permuted Canonical Constraint Order: {final_constr_order}")
            self.logger.lazy_debug(f"Original Canonical Variable Order: {original_canonical_var_order}")
            self.logger.lazy_debug(f"Permuted Canonical Variable Order: {final_var_order}")

            canonical_distance = self.permutator.permutation_distance(
                original_canonical_constr_order, original_canonical_var_order,
                final_constr_order, final_var_order,
                self.row_distance_metric,
                self.col_distance_metric,
                alpha=1.0, 
                beta=1.0,
            )
            self.logger.info(f"Permutation Distance After Canonicalization: {canonical_distance}")

            # === 4. Apply final canonical ordering to the unscaled permuted model and solve it ===
            self.logger.info("Applying final canonical ordering to permuted model and solving it")
            ordered_permuted_model = self.permutator.apply_permutation(permuted_model, permuted_canonical_var_order, permuted_canonical_constr_order)
            if LOG_MATRIX:
                self.canonical_matrices.append(ordered_permuted_model.getA())
            
            self.logger.info("Solving Reordering Form from Permuted Model")
            final_ordered_result = self.solve_problem(ordered_permuted_model)
            #ProblemPrinterlog_model(ordered_permuted_model, self.logger, level="DEBUG")

            return permuted_canonical, permuted_result, final_ordered_result, permuted_distance, canonical_distance

        except Exception as e:
            self.logger.error(f"Error in single iteration: {str(e)}")
            raise
    
    def build_iteration_result(self, original_canonical, permuted_canonical, original_result, permuted_result, canonical_from_original_result, final_ordered_result, permuted_distance, canonical_distance):
        """
        Constructs the iteration result dictionary from the solved models.
        """
        # Validate equivalence between canonical forms (from original and from permuted)
        are_equivalent = self.canonical_generator.validate(original_canonical, permuted_canonical)

        if LOG_MODEL_COMPARISON and not are_equivalent:
            IterationLogger().log_model_comparison(original_canonical, permuted_canonical)

        return {
            'equivalent': are_equivalent,
            'original_vars': original_canonical.NumVars,
            'permuted_vars': permuted_canonical.NumVars,
            'original_constrs': original_canonical.NumConstrs,
            'permuted_constrs': permuted_canonical.NumConstrs,
            'original_objective': original_result['objective_value'],
            'permuted_objective': permuted_result['objective_value'],
            'original_solve_time': original_result['solve_time'],
            'permuted_solve_time': permuted_result['solve_time'],
            'original_work_units': original_result['work_units'],
            'permuted_work_units': permuted_result['work_units'],
            'canonical_from_original_objective': canonical_from_original_result['objective_value'],
            'canonical_from_permuted_objective': final_ordered_result['objective_value'],
            'canonical_from_original_solve_time': canonical_from_original_result['solve_time'],
            'canonical_from_permuted_solve_time': final_ordered_result['solve_time'],
            'canonical_from_original_work_units': canonical_from_original_result['work_units'],
            'canonical_from_permuted_work_units': final_ordered_result['work_units'],
            'permutation_distance_before_canonicalization': permuted_distance,
            'permutation_distance_after_canonicalization': canonical_distance,
        }
            

    def load_problem(self):
        """
        Load the optimization problem from LP/MPS file, either locally or from S3.
        """
        if PRODUCTION:
            import boto3
            from botocore.exceptions import ClientError

            if not self.file_path:
                raise ValueError("File path cannot be empty in production mode.")
            key = f"{self.file_path}"
            bucket_name = BUCKET_NAME
            try:
                s3 = boto3.client('s3')
                response = s3.get_object(Bucket=bucket_name, Key=key)
                file_content = response['Body'].read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mps") as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                model = gp.read(temp_file_path, env=self.gp_env)
                self.logger.info(f"Successfully loaded problem from S3: s3://{bucket_name}/{key}")
                os.remove(temp_file_path)
                return model
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    raise FileNotFoundError(f"The file s3://{bucket_name}/{key} does not exist in S3")
                raise
        else:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"The file {self.file_path} does not exist locally")
            model = gp.read(self.file_path, env=self.gp_env)
            self.logger.info(f"Successfully loaded local problem from {self.file_path}")
            return model

    def solve_problem(self, model):
        """
        Solve the given optimization problem.
        """
        print(model)
        if DISABLE_SOLVING:
            self.logger.info("Solving is disabled via environment variable. Returning placeholder result.")
            return {
                "solve_status": "Solving Disabled",
                "status_message": "Solving was skipped",
                "objective_value": None,
                "solution": None,
                "solve_time": 0,
                "work_units": 0
            }

        try:
            model.Params.Threads = NUMBER_OF_THREADS
            model.Params.TimeLimit = MAX_SOLVE_TIME
            start_time = time.time()
            model.optimize()
            elapsed_time = time.time() - start_time
            status = model.Status
            status_map = {
                GRB.OPTIMAL: "Optimal solution found",
                GRB.INFEASIBLE: "Problem is infeasible",
                GRB.UNBOUNDED: "Problem is unbounded",
                GRB.INF_OR_UNBD: "Problem is infeasible or unbounded",
                GRB.INTERRUPTED: "Optimization was interrupted",
            }
            result = {
                "solve_status": status,
                "status_message": status_map.get(status, "Unknown status"),
                "objective_value": model.ObjVal if status == GRB.OPTIMAL else None,
                "solution": model.getAttr('X', model.getVars()) if status == GRB.OPTIMAL else None,
                "solve_time": elapsed_time,
                "work_units" : model.work
            }
            self.logger.info(f"Solve Status: {result['status_message']}")
            self.logger.info(f"Solve Time: {elapsed_time:.4f} seconds")
            if status == GRB.OPTIMAL:
                self.logger.info(f"Objective Value: {result['objective_value']}")
            return result
        except gp.GurobiError as e:
            self.logger.error(f"Gurobi Error: {e}")
            raise
        
