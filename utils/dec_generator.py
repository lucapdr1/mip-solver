import os
import uuid
import shutil
import tempfile
import subprocess
import gurobipy as gp

def generate_unique_id() -> str:
    return uuid.uuid4().hex[:8]

def create_cmd_file(temp_input: str, output_dir: str, unique_id: str) -> str:
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

def run_gcg(cmd_file: str, output_dir: str, timeout: int = 300):
    subprocess.run(
        ["gcg", "-b", cmd_file],
        cwd=output_dir,
        check=True,
        timeout=timeout
    )

def find_latest_dec_file(output_dir: str) -> str:
    dec_files = [f for f in os.listdir(output_dir) if f.endswith('.dec')]
    if not dec_files:
        raise FileNotFoundError("GCG did not produce a .dec file.")
    dec_files = sorted(dec_files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)), reverse=True)
    return os.path.join(output_dir, dec_files[0])

def preprocess_with_gcg(model, gp_env, dec_output_path: str = None):
    """
    Preprocesses the model using GCG and saves the decomposition (.dec) file if specified.

    Args:
        model: The Gurobi model to preprocess.
        gp_env: Gurobi environment (unused in this modification but kept for compatibility).
        dec_output_path: Optional; full path to save the generated .dec file.
    
    Returns:
        constr_perm: Constraint permutation derived from the decomposition.
    """
    unique_id = generate_unique_id()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_input = os.path.join(tmp_dir, f"temp_input_{unique_id}.mps")
        
        # Write the model to a temporary MPS file
        model.write(temp_input)
        
        # Create and run GCG command file
        cmd_file = create_cmd_file(temp_input, tmp_dir, unique_id)
        run_gcg(cmd_file, tmp_dir)
        
        # Locate and save the .dec file if requested
        dec_file_path = find_latest_dec_file(tmp_dir)
        if dec_output_path:
            output_dir = os.path.dirname(dec_output_path) or '.'
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(dec_file_path, dec_output_path)
            print(f"✓ Decomposition saved to {os.path.abspath(dec_output_path)}")