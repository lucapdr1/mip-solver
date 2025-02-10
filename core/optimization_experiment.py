import os
import time
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
import tempfile
from utils.logging_handler import LoggingHandler
from core.problem_permutator import ProblemPermutator
from core.canonical_form_generator import CanonicalFormGenerator
from core.problem_normalizer import Normalizer
from utils.problem_printer import ProblemPrinter
from utils.config import LOG_MODEL_COMPARISON, PRODUCTION, BUCKET_NAME

class OptimizationExperiment:
    def __init__(self, gp_env, file_path, ordering_rule):
        self.gp_env = gp_env
        self.file_path = file_path
        self.logger = LoggingHandler().get_logger()
        self.normalizer = Normalizer()
        self.original_model = self.load_problem()
        self.ordering_rule = ordering_rule
        
        self.logger.info(f"Successfully loaded problem from {file_path}")
        self.logger.info("Problem Details:")
        self.logger.info(f"- Variables: {self.original_model.NumVars}")
        self.logger.info(f"- Constraints: {self.original_model.NumConstrs}")
        self.logger.info(f"- Objective Sense: {'Minimize' if self.original_model.ModelSense == 1 else 'Maximize'}")
        
        self.permutator = ProblemPermutator(gp_env, self.original_model)
        self.canonical_generator = CanonicalFormGenerator(gp_env, self.original_model, self.ordering_rule, self.normalizer)

    def run_single_iteration(self, original_result, canonical_from_original_result, original_canonical, original_canonical_var_order, original_canonical_constr_order):
        """Run a single iteration of the experiment with solving and detailed logging"""
        try:
            self.logger.debug("Starting new iteration...")

            # Create permuted problem
            self.logger.info("Creating Permuted Problem")
            permuted_model, var_permutation, constr_permutation, _, _ = self.permutator.create_permuted_problem()

            ProblemPrinter.log_model(permuted_model, self.logger, level="DEBUG")

            # Compute permutation distance BEFORE canonicalization
            self.logger.info("Computing Permutation Distance before Canonicalization...")

            original_var_order = list(range(self.original_model.NumVars))
            original_constr_order = list(range(self.original_model.NumConstrs))

            self.logger.debug(f"Original Constraint Order: {original_constr_order}")
            self.logger.debug(f"Permuted Constraint Order: {constr_permutation}")
            self.logger.debug(f"Original Variable Order: {original_var_order}")
            self.logger.debug(f"Permuted Variable Order: {var_permutation}")
          
            permuted_distance = self.permutator.permutation_distance(
                original_constr_order,  original_var_order,
                constr_permutation, var_permutation,
                row_dist_method="kendall_tau",
                col_dist_method="kendall_tau",
                alpha=1.0, 
                beta=1.0
            )

            self.logger.info(f"Permutation Distance Before Canonicalization: {permuted_distance}")

            # Solve the permuted problem
            self.logger.info("Solving Permuted Problem")
            permuted_result = self.solve_problem(permuted_model)

            # Compare original and permuted before canonicalization
            self.logger.debug("Before canonicalization:")
            A_orig = self.original_model.getA()
            A_perm = permuted_model.getA()
            self.logger.debug(f"Original matrix: shape={A_orig.shape}, nnz={A_orig.nnz}")
            self.logger.debug(f"Permuted matrix: shape={A_perm.shape}, nnz={A_perm.nnz}")

            # Generate canonical form for the permuted model
            self.logger.debug("Generating canonical form for permuted model...")
            permuted_canonical, permuted_canonical_var_order, permuted_canonical_constr_order = (
                CanonicalFormGenerator(self.gp_env, permuted_model, self.ordering_rule, self.normalizer)
                .get_canonical_form()
            )

            ProblemPrinter.log_model(permuted_canonical, self.logger, level="DEBUG")

            self.logger.info("Solving Canonical Form from Permuted Model")
            canonical_from_permuted_result = self.solve_problem(permuted_canonical)

            # Compute COMPOSED permutation order
            self.logger.info("Computing Adjusted Permuted Canonical Order...")
            
            # Correct permutation composition: π_final = π_perm ∘ π_canon (applied on original)
            final_var_order = var_permutation[permuted_canonical_var_order]
            final_constr_order = constr_permutation[permuted_canonical_constr_order]


            # Compute permutation distance AFTER canonicalization
            self.logger.info("Computing Permutation Distance after Canonicalization...")

            self.logger.debug(f"Original Canonical Constraint Order: {original_canonical_constr_order}")
            self.logger.debug(f"Permuted Canonical Constraint Order: {final_constr_order}")
            self.logger.debug(f"Original Canonical Variable Order: {original_canonical_var_order}")
            self.logger.debug(f"Permuted Canonical Variable Order: {final_var_order}")

            canonical_distance = self.permutator.permutation_distance(
                original_canonical_constr_order, original_canonical_var_order,
                final_constr_order, final_var_order,
                row_dist_method="kendall_tau",
                col_dist_method="kendall_tau",
                alpha=1.0, 
                beta=1.0
            )

            self.logger.info(f"Permutation Distance After Canonicalization: {canonical_distance}")

            # Validate equivalence
            are_equivalent = self.canonical_generator.validate(original_canonical, permuted_canonical)

            # Store results, including permutation distances
            result = {
                'equivalent': are_equivalent,
                'original_vars': original_canonical.NumVars,
                'permuted_vars': permuted_canonical.NumVars,
                'original_constrs': original_canonical.NumConstrs,
                'permuted_constrs': permuted_canonical.NumConstrs,
                'original_objective': original_result['objective_value'],
                'permuted_objective': permuted_result['objective_value'],
                'original_solve_time': original_result['solve_time'],
                'permuted_solve_time': permuted_result['solve_time'],
                'canonical_from_original_objective': canonical_from_original_result['objective_value'],
                'canonical_from_permuted_objective': canonical_from_permuted_result['objective_value'],
                'canonical_from_original_solve_time': canonical_from_original_result['solve_time'],
                'canonical_from_permuted_solve_time': canonical_from_permuted_result['solve_time'],
                'permutation_distance_before_canonicalization': permuted_distance,
                'permutation_distance_after_canonicalization': canonical_distance
            }

            # Log detailed differences if not equivalent
            if LOG_MODEL_COMPARISON and not are_equivalent:
                self.logger.debug("Detailed model comparison:")
                LoggingHandler().log_model_differences(self.logger, original_canonical, permuted_canonical)

            return result

        except Exception as e:
            self.logger.error(f"Error in single iteration: {str(e)}")
            raise

    def run_experiment(self, num_iterations):
        """Run multiple iterations with detailed logging and solving functionality"""
        ProblemPrinter.log_model(self.original_model, self.logger, level="DEBUG")
        results = []

        # Solve the original problem once
        self.logger.info("Solving Original Problem")
        original_result = self.solve_problem(self.original_model)

        # Generate the canonical form of the original model once
        self.logger.debug("Generating canonical form for the original model...")
        original_canonical, original_canonical_var_order, original_canonical_constr_order = self.canonical_generator.get_canonical_form()
        ProblemPrinter.log_model(original_canonical, self.logger, level="DEBUG")

        # Solve the Canonical from original once
        self.logger.info("Solving Canonical from Original Problem")
        canonical_from_original_result  = self.solve_problem(original_canonical)

        for i in range(num_iterations):
            self.logger.info(f"Running iteration {i+1}/{num_iterations}")
            try:
                iteration_result = self.run_single_iteration(original_result, canonical_from_original_result ,original_canonical, original_canonical_var_order, original_canonical_constr_order)
                results.append(iteration_result)
                self.log_iteration_results(i + 1, iteration_result)

            except Exception as e:
                self.logger.error(f"Error in iteration {i+1}: {str(e)}")
                raise

        # Compute and log solve-time variability metrics
        self.compute_solve_time_variability(results)
        return results

    def load_problem(self):
        """
        Load the optimization problem from LP/MPS file, either locally or from S3
        based on production flag.

        Returns:
            gurobipy.Model: Loaded Gurobi model
        """
        if PRODUCTION:
            # Handle S3 path (self.file_path is the part after input/)
            if not self.file_path:
                raise ValueError("File path cannot be empty in production mode.")
            
            # Construct the full S3 key by prepending "input/"
            key = f"{self.file_path}"
            bucket_name = BUCKET_NAME  # Your bucket name
            
            try:
                # Create S3 client
                s3 = boto3.client('s3')
                
                # Get object from S3
                response = s3.get_object(Bucket=bucket_name, Key=key)
                file_content = response['Body'].read()
                
                ## Save the content to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mps") as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                # Load model from the temporary file
                model = gp.read(temp_file_path, env=self.gp_env)
                self.logger.info(f"Successfully loaded problem from S3: s3://{bucket_name}/{key}")
                
                # Clean up the temporary file
                os.remove(temp_file_path)
                return model
                
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    raise FileNotFoundError(f"The file s3://{bucket_name}/{key} does not exist in S3")
                raise
                
        else:
            # Local file handling
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"The file {self.file_path} does not exist locally")
                
            model = gp.read(self.file_path, env=self.gp_env)
            self.logger.info(f"Successfully loaded local problem from {self.file_path}")
            return model
        
    def solve_problem(self, model):
        """
        Solve the given optimization problem.

        Args:
            model (gurobipy.Model): Gurobi model to solve

        Returns:
            dict: Solving results
        """
        try:
            start_time = time.time()  # Start the timer
            model.optimize()
            elapsed_time = time.time() - start_time  # Calculate elapsed time

            status = model.Status

            # Comprehensive status handling
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
                "solve_time": elapsed_time,  # Include the solve time
            }

            self.logger.info(f"Solve Status: {result['status_message']}")
            self.logger.info(f"Solve Time: {elapsed_time:.4f} seconds")  # Log the solve time
            if status == GRB.OPTIMAL:
                self.logger.info(f"Objective Value: {result['objective_value']}")

            return result

        except gp.GurobiError as e:
            self.logger.error(f"Gurobi Error: {e}")
            raise

    def log_iteration_results(self, iteration_num, iteration_result):
        """Logs the results of a single iteration"""
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

    def compute_solve_time_variability(self, results):
        """
        Computes and logs the variability in solve times for:
        - Original vs Permuted model
        - Canonical (from Original) vs Canonical (from Permuted)
        """
        import numpy as np

        def relative_difference(x, y):
            # Small epsilon to avoid division by zero
            return abs(x - y) / max(abs(x), abs(y), 1e-12)

        # Collect the relative differences in solve times
        original_vs_permuted_diffs = []
        canonical_vs_canonical_diffs = []

        for res in results:
            original_vs_permuted_diffs.append(
                relative_difference(res["original_solve_time"], res["permuted_solve_time"])
            )
            canonical_vs_canonical_diffs.append(
                relative_difference(
                    res["canonical_from_original_solve_time"],
                    res["canonical_from_permuted_solve_time"]
                )
            )

        # Compute mean solve-time variability
        mean_orig_perm_var = np.mean(original_vs_permuted_diffs) if original_vs_permuted_diffs else 0.0
        mean_canon_var = np.mean(canonical_vs_canonical_diffs) if canonical_vs_canonical_diffs else 0.0

        self.logger.info("Solve-Time Variability Metrics:")
        self.logger.info(f"- Mean Solve-Time Variability (Original vs Permuted): {mean_orig_perm_var:.6f}")
        self.logger.info(f"- Mean Solve-Time Variability (Canonical vs Canonical): {mean_canon_var:.6f}")

        if mean_canon_var < mean_orig_perm_var:
            self.logger.info("Canonical form has improved solve-time consistency across permutations.")
        else:
            self.logger.warning("Canonical form still exhibits significant solve-time variability.")


