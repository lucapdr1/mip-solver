import os
import gurobipy as gp
import numpy as np
import tempfile

from core.problem_transform.problem_permutator import ProblemPermutator
from utils.config import PERMUTE_SEED, PERMUTE_GRANULARITY_K
from utils.dec_generator import preprocess_with_gcg

class DecGenerator:
    def __init__(self, gp_env, file_path):
        self.gp_env = gp_env
        self.file_path = file_path
        self.original_model = None
        self.permutator = None

    def load_problem(self):
        """Local file loading implementation"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Local file {self.file_path} not found")
        return gp.read(self.file_path, env=self.gp_env)

    def process_permutations(self, num_permutations, output_dir="decompositions"):
        """
        Generate permuted models with decomposition files
        Uses environment variables for configuration:
        - PERMUTE_GRANULARITY_K: Block size for permutations
        - PERMUTE_SEED: Base seed for random operations
        - APPLY_DEC: Whether to generate decomposition files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(num_permutations):
            # Create permuted model with configured parameters
            permuted_model, var_perm, constr_perm, _, _ = self.permutator.create_permuted_problem(
                PERMUTE_GRANULARITY_K,
                PERMUTE_SEED + i  # Maintain seed sequence from original
            )
            dec_path = os.path.join(output_dir, f"perm_{PERMUTE_SEED+i}.dec")
            self._generate_decomposition(permuted_model, dec_path)

    def _generate_decomposition(self, model, dec_output_path):
        """Internal method handling DEC file generation"""
        preprocess_with_gcg(
            model=model,
            gp_env=self.gp_env,
            dec_output_path=dec_output_path
        )

    def create_decmpositions(self, num_of_permutations):
        """Execute full loading and preprocessing pipeline"""
        self.original_model = self.load_problem()
        self.permutator = ProblemPermutator(self.gp_env, self.original_model)
        self.process_permutations(num_permutations=num_of_permutations)