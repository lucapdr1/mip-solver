# core/problem_permutator.py

import numpy as np
import xpress as xp
from core.logging_handler import LoggingHandler

class ProblemPermutator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.logger = LoggingHandler().get_logger()

    def create_permuted_problem(self):
        """
        Create a randomly permuted version of the original problem.
        
        Returns:
            xpress.problem: Permuted Xpress problem
        """
        # Create a copy of the original model
        permuted_model = xp.problem()
        permuted_model.read(self.file_path)
        
        # Get number of columns (variables)
        num_cols = permuted_model.attributes.cols
        
        # Create a random permutation
        permutation = np.random.permutation(num_cols)
        
        self.logger.info("Creating permuted problem:")
        self.logger.info(f"Permutation: {permutation}")
        
        return permuted_model
