import os
import os
import uuid
import shutil
import tempfile
import subprocess
import gurobipy as gp

from core.problem_transform.problem_permutator import ProblemPermutator
from utils.config import PERMUTE_SEED, PERMUTE_GRANULARITY_K, OUTPUT_DIR
from utils.logging_handler import LoggingHandler

class DecGenerator:
    def __init__(self, gp_env, file_path):
        self.gp_env = gp_env
        self.file_path = file_path
        self.original_model = None
        self.permutator = None
        self.logger = LoggingHandler().get_logger()
        # Extract base name from the file path
        self.base_name = os.path.splitext(os.path.basename(file_path))[0]

    def load_problem(self):
        """Local file loading implementation"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Local file {self.file_path} not found")
        return gp.read(self.file_path, env=self.gp_env)

    def process_permutations(self, num_permutations):
        """
        Generate permuted models with decomposition files
        Uses environment variables for configuration:
        - PERMUTE_GRANULARITY_K: Block size for permutations
        - PERMUTE_SEED: Base seed for random operations
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        for i in range(num_permutations):
            # Create permuted model with configured parameters
            permuted_model, var_perm, constr_perm, _, _ = self.permutator.create_permuted_problem(
                PERMUTE_GRANULARITY_K,
                PERMUTE_SEED + i  # Maintain seed sequence from original
            )
            # Generate decomposition file name based on original problem name and permutation index
            dec_path = os.path.join(OUTPUT_DIR, f"{self.base_name}_{i}_{PERMUTE_GRANULARITY_K}.dec")
            self._generate_decomposition(permuted_model, dec_path)

    def _generate_decomposition(self, model, dec_output_path):
        """Internal method handling DEC file generation"""
        self.preprocess_with_gcg(
            model=model,
            gp_env=self.gp_env,
            dec_output_path=dec_output_path
        )

    def create_decmpositions(self, num_of_permutations):
        """Execute full loading and preprocessing pipeline"""
        self.original_model = self.load_problem()
        self.permutator = ProblemPermutator(self.gp_env, self.original_model)
        self.process_permutations(num_permutations=num_of_permutations)

    def generate_unique_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def create_cmd_file(self,temp_input: str, output_dir: str, unique_id: str) -> str:
        cmd_file = os.path.join(output_dir, f"cmds_{unique_id}.cmd")
        with open(cmd_file, 'w') as f:
            f.write(f'read "{temp_input}"\n')
            f.write('set detection enabled TRUE\n')
            f.write('detect\n')
            f.write('explore\n')
            f.write('score 5\n')  # max-white score
            f.write('select 0\n')
            f.write('export 0\n')
            f.write('quit\n')
            f.write(f'write selected "{output_dir}" dec\n')
            f.write('quit\n')
        return cmd_file

    def run_gcg(self, cmd_file: str, output_dir: str, timeout: int = 300):
        subprocess.run(
            ["gcg", "-b", cmd_file],
            cwd=output_dir,
            check=True,
            timeout=timeout
        )

    def find_latest_dec_file(self, output_dir: str) -> str:
        dec_files = [f for f in os.listdir(output_dir) if f.endswith('.dec')]
        if not dec_files:
            raise FileNotFoundError("GCG did not produce a .dec file.")
        dec_files = sorted(dec_files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)), reverse=True)
        return os.path.join(output_dir, dec_files[0])

    def preprocess_with_gcg(self, model, gp_env, dec_output_path: str = None):
        """
        Preprocesses the model using GCG and saves the decomposition (.dec) file if specified.

        Args:
            model: The Gurobi model to preprocess.
            gp_env: Gurobi environment (unused in this modification but kept for compatibility).
            dec_output_path: Optional; full path to save the generated .dec file.
        
        Returns:
            constr_perm: Constraint permutation derived from the decomposition.
        """
        unique_id = self.generate_unique_id()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_input = os.path.join(tmp_dir, f"temp_input_{unique_id}.mps")
            
            # Write the model to a temporary MPS file
            model.write(temp_input)
            
            # Create and run GCG command file
            cmd_file = self.create_cmd_file(temp_input, tmp_dir, unique_id)
            self.run_gcg(cmd_file, tmp_dir)
            
            # Locate and save the .dec file if requested
            dec_file_path = self.find_latest_dec_file(tmp_dir)
            if dec_output_path:
                output_dir = os.path.dirname(dec_output_path) or '.'
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy(dec_file_path, dec_output_path)
                print(f"✓ Decomposition saved to {os.path.abspath(dec_output_path)}")