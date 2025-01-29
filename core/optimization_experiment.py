import os
import time
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from utils.logging_handler import LoggingHandler
from core.problem_permutator import ProblemPermutator
from core.canonical_form_generator import CanonicalFormGenerator
from core.problem_normalizer import Normalizer
from utils.problem_printer import ProblemPrinter
from utils.config import LOG_MODEL_COMPARISON

class OptimizationExperiment:
    def __init__(self, file_path, ordering_rule):
        self.file_path = file_path
        self.logger = LoggingHandler().get_logger()
        self.normalizer = Normalizer()
        self.original_model = gp.read(file_path)
        self.ordering_rule = ordering_rule
        
        self.logger.info(f"Successfully loaded problem from {file_path}")
        self.logger.info("Problem Details:")
        self.logger.info(f"- Variables: {self.original_model.NumVars}")
        self.logger.info(f"- Constraints: {self.original_model.NumConstrs}")
        self.logger.info(f"- Objective Sense: {'Minimize' if self.original_model.ModelSense == 1 else 'Maximize'}")
        
        self.permutator = ProblemPermutator(file_path)
        self.canonical_generator = CanonicalFormGenerator(self.original_model, self.ordering_rule, self.normalizer)

    def run_single_iteration(self, original_result, original_canonical):
        """Run a single iteration of the experiment with solving and detailed logging"""
        try:
            self.logger.debug("Starting new iteration...")

            # Create permuted problem
            self.logger.info("Creating Permuted Problem")
            permuted_model = self.permutator.create_permuted_problem()
            ProblemPrinter.log_model(permuted_model, self.logger, level="DEBUG")

            # Solve the permuted problem
            self.logger.info("Solving Permuted Problem")
            permuted_result = self.solve_problem(permuted_model)

            # Log permuted problem details
            self.logger.info("Permuted Problem Results:")
            self.logger.info(f"- Objective Value: {permuted_result['objective_value']}")
            self.logger.info(f"- Solve Time: {permuted_result['solve_time']:.10f} seconds")

            # Compare original and permuted before canonicalization
            self.logger.debug("Before canonicalization:")
            A_orig = self.original_model.getA()
            A_perm = permuted_model.getA()
            self.logger.debug(f"Original matrix: shape={A_orig.shape}, nnz={A_orig.nnz}")
            self.logger.debug(f"Permuted matrix: shape={A_perm.shape}, nnz={A_perm.nnz}")

            # Generate canonical form for the permuted model
            self.logger.debug("Generating canonical form for permuted model...")
            permuted_canonical = CanonicalFormGenerator(permuted_model, self.ordering_rule, self.normalizer).get_canonical_form()
            ProblemPrinter.log_model(permuted_canonical, self.logger, level="DEBUG")

            # Solve the canonical forms
            self.logger.info("Solving Canonical Form from Original Model")
            canonical_from_original_result = self.solve_problem(original_canonical)

            self.logger.info("Solving Canonical Form from Permuted Model")
            canonical_from_permuted_result = self.solve_problem(permuted_canonical)

            # Validate equivalence
            are_equivalent = self.canonical_generator.validate(original_canonical, permuted_canonical)

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

        # Log original problem details
        self.logger.info("Original Problem Results:")
        self.logger.info(f"- Objective Value: {original_result['objective_value']}")
        self.logger.info(f"- Solve Time: {original_result['solve_time']:.10f} seconds")

        # Generate the canonical form of the original model once
        self.logger.debug("Generating canonical form for the original model...")
        original_canonical = self.canonical_generator.get_canonical_form()
        ProblemPrinter.log_model(original_canonical, self.logger, level="DEBUG")

        for i in range(num_iterations):
            self.logger.info(f"Running iteration {i+1}/{num_iterations}")
            try:
                iteration_result = self.run_single_iteration(original_result, original_canonical)
                results.append(iteration_result)

                # Log iteration results
                self.logger.info(f"Iteration {i+1} Results:")
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

            except Exception as e:
                self.logger.error(f"Error in iteration {i+1}: {str(e)}")
                raise

        return results

    def _load_problem(self):
        """
        Load the optimization problem from LP or MPS file.

        Returns:
            gurobipy.Model: Loaded Gurobi model
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        model = gp.read(self.file_path)
        self.logger.info(f"Successfully loaded problem from {self.file_path}")
        self.logger.info(f"Problem Details:")
        self.logger.info(f"- Variables: {model.NumVars}")
        self.logger.info(f"- Constraints: {model.NumConstrs}")
        self.logger.info(f"- Objective Sense: {'Minimize' if model.ModelSense == 1 else 'Maximize'}")

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


