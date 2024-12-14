# core/optimization_experiment.py

import os
import xpress as xp
from core.logging_handler import LoggingHandler
from core.problem_permutator import ProblemPermutator

class OptimizationExperiment:
    def __init__(self, file_path, log_dir='experiments'):
        """
        Initialize the optimization experiment.
        
        Args:
            file_path (str): Path to the MPS file
            log_dir (str): Directory to store experiment logs
        """
        # Initialize the logger
        self.logger = LoggingHandler(log_dir).get_logger()
        
        self.file_path = file_path
        self.original_model = self._load_problem()

    def _load_problem(self):
        """
        Load the optimization problem from MPS file.
        
        Returns:
            xpress.problem: Loaded Xpress problem
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        
        model = xp.problem()
        model.read(self.file_path)
        
        self.logger.info(f"Successfully loaded problem from {self.file_path}")
        self.logger.info(f"Problem Details:")
        self.logger.info(f"- Variables: {model.attributes.cols}")
        self.logger.info(f"- Constraints: {model.attributes.rows}")
        self.logger.info(f"- Objective Sense: {'Minimize' if model.attributes.objsense == 1 else 'Maximize'}")
        
        return model

    def solve_problem(self, model):
        """
        Solve the given optimization problem.
        
        Args:
            model (xpress.problem): Xpress problem to solve
        
        Returns:
            dict: Solving results
        """
        try:
            model.solve()
            
            solve_status = model.attributes.solvestatus
            sol_status = model.attributes.solstatus
            
            # Comprehensive status handling
            status_map = {
                0: "Optimal solution found",
                1: "Problem is infeasible",
                2: "Problem is unbounded",
                3: "Problem has no solution",
                4: "Solution is integer infeasible"
            }
            
            result = {
                "solve_status": solve_status,
                "sol_status": sol_status,
                "status_message": status_map.get(solve_status, "Unknown status"),
                "objective_value": model.getObjVal() if solve_status == 0 else None,
                "solution": model.getSolution() if solve_status == 0 else None
            }
            
            self.logger.info(f"Solve Status: {result['status_message']}")
            if solve_status == 0:
                self.logger.info(f"Objective Value: {result['objective_value']}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error solving problem: {e}")
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
        
        if original_result['solve_status'] == 0 and permuted_result['solve_status'] == 0:
            objective_diff = abs(original_result['objective_value'] - permuted_result['objective_value'])
            relative_diff = objective_diff / abs(original_result['objective_value']) * 100
            
            self.logger.info(f"Absolute Difference: {objective_diff}")
            self.logger.info(f"Relative Difference: {relative_diff:.4f}%")
        
        self.logger.info("Experiment Completed")
        
        return {
            'original_result': original_result,
            'permuted_result': permuted_result,
        }
