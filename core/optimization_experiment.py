# core/optimization_experiment.py

import os
import gurobipy as gp
from gurobipy import GRB
from core.logging_handler import LoggingHandler
from core.problem_permutator import ProblemPermutator

class OptimizationExperiment:
    def __init__(self, file_path, log_dir='experiments'):
        """
        Initialize the optimization experiment.
        
        Args:
            file_path (str): Path to the LP or MPS file
            log_dir (str): Directory to store experiment logs
        """
        # Initialize the logger
        self.logger = LoggingHandler(log_dir).get_logger()
        
        self.file_path = file_path
        self.original_model = self._load_problem()

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
            model.optimize()

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
                "solution": model.getAttr('X', model.getVars()) if status == GRB.OPTIMAL else None
            }

            self.logger.info(f"Solve Status: {result['status_message']}")
            if status == GRB.OPTIMAL:
                self.logger.info(f"Objective Value: {result['objective_value']}")

            return result

        except gp.GurobiError as e:
            self.logger.error(f"Gurobi Error: {e}")
            raise

    def run_experiment(self):
        """
        Run the complete optimization experiment.
        Compare original and permuted problem solutions.
        """
        self.logger.info("Starting Optimization Experiment")
        
        # Solve original problem
        self.logger.info("Solving Original Problem")
        original_result = self.solve_problem(self.original_model)
        
        # Create and solve permuted problem
        self.logger.info("Creating Permuted Problem")
        permuted_model = ProblemPermutator(self.file_path).create_permuted_problem()
        
        self.logger.info("Solving Permuted Problem")
        permuted_result = self.solve_problem(permuted_model)
        
        # Compare results
        self.logger.info("Comparing Results")
        self.logger.info(f"Original Objective: {original_result['objective_value']}")
        self.logger.info(f"Permuted Objective: {permuted_result['objective_value']}")
        
        if original_result['solve_status'] == GRB.OPTIMAL and permuted_result['solve_status'] == GRB.OPTIMAL:
            objective_diff = abs(original_result['objective_value'] - permuted_result['objective_value'])
            relative_diff = objective_diff / abs(original_result['objective_value']) * 100
            
            self.logger.info(f"Absolute Difference: {objective_diff}")
            self.logger.info(f"Relative Difference: {relative_diff:.4f}%")
        
        self.logger.info("Experiment Completed")
        
        return {
            'original_result': original_result,
            'permuted_result': permuted_result,
        }

