import os
import shutil
import subprocess
import uuid
import gurobipy as gp
from datetime import datetime
from utils.gcg_dec_parser import constraint_permutation_from_dec
from core.problem_transform.problem_permutator import ProblemPermutator

def process_gp_file(output_dir: str, base_name: str) -> None:
    """Process .gp file and handle GCG's auto-generated PDF output."""
    try:
        # 1. Find and rename the GCG-generated .gp file
        gp_files = [f for f in os.listdir(output_dir) 
                   if f.endswith('.gp')]
        
        if not gp_files:
            print("No GCG-generated .gp file found")
            return

        original_gp = os.path.join(output_dir, gp_files[0])
        clean_gp = os.path.join(output_dir, f"{base_name}_decomposition.gp")
        os.rename(original_gp, clean_gp)
        
        # 2. Generate the visualization (GCG will create its own PDF)
        subprocess.run(
            ["gnuplot", clean_gp],
            cwd=output_dir,
            check=True
        )
        
        # 3. Find and rename GCG's auto-generated PDF
        pdf_files = [f for f in os.listdir(output_dir) 
                    if f.endswith('.pdf') and f.startswith('_MPS_')]
        
        if pdf_files:
            original_pdf = os.path.join(output_dir, pdf_files[0])
            final_pdf = os.path.join(output_dir, f"{base_name}_decomposition.pdf")
            os.rename(original_pdf, final_pdf)
            print(f"✓ PDF visualization saved to {final_pdf}")
        else:
            print("Warning: No PDF output was generated")
        
        # 4. Clean up temporary files
        os.remove(clean_gp)
        
        # Remove any other GCG-generated plot files
        for f in os.listdir(output_dir):
            if f.startswith('_MPS_') and (f.endswith('.eps') or f.endswith('.gp')):
                os.remove(os.path.join(output_dir, f))

    except Exception as e:
        print(f"Error processing visualization: {str(e)}")
        # Clean up any partial files
        for f in ['original_gp', 'clean_gp', 'original_pdf']:
            if f in locals() and os.path.exists(eval(f)):
                os.remove(eval(f))


def gcg_preprocess(gp_env: gp.Env, input_file: str, output_file: str = None) -> gp.Model:
    """Robust GCG preprocessing with strict file handling.
    This version runs GCG (which now writes a .dec file), then uses a permutation
    (derived from the dec file) to reorder the original model's constraints.
    The final permuted model is returned as a gurobipy.Model.
    """
    # Generate unique ID upfront
    unique_id = uuid.uuid4().hex[:8]
    
    try:
        input_file = os.path.abspath(input_file)
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_dir = f"{base}_gcg_output_{unique_id}"
            os.makedirs(output_dir, exist_ok=True)
            # We still define an output file name for temporary writing if needed.
            output_file = os.path.join(output_dir, f"structured_{unique_id}{ext}")
        else:
            output_file = os.path.abspath(output_file)
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)

        # Create a temporary copy of the input file in the output directory.
        temp_input = os.path.join(output_dir, f"temp_input_{unique_id}.mps")
        shutil.copy2(input_file, temp_input)

        # Create a unique command file for GCG.
        cmd_file = os.path.join(output_dir, f"cmds_{unique_id}.cmd")
        with open(cmd_file, 'w') as f:
            f.write(f'read "{temp_input}"\n')
            f.write('set detection enabled TRUE\n')
            f.write('detect\n')
            f.write('explore\n')
            f.write('select 0\n')
            f.write('export 0\n')
            f.write('quit\n')
            # Write the dec file instead of a PDF: note "dec" at the end.
            f.write(f'write selected "{output_dir}" dec\n')
            f.write('quit\n')

        # Run GCG with timeout.
        subprocess.run(
            ["gcg", "-b", cmd_file],
            cwd=output_dir,
            check=True,
            timeout=300  # 5 minutes timeout
        )
        
        # Optionally, process visualization.
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        process_gp_file(output_dir, base_name)  # if needed
        
        # Look for the generated .dec file in the output directory.
        dec_files = [f for f in os.listdir(output_dir) if f.endswith('.dec')]
        if not dec_files:
            raise FileNotFoundError("GCG did not produce a .dec file.")
        # Use the most recently created .dec file.
        dec_files = sorted(dec_files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)), reverse=True)
        dec_file_path = os.path.join(output_dir, dec_files[0])
        
        # Load the temporary model from temp_input.
        original_model = gp.read(temp_input)
        
        # Compute the constraint permutation from the dec file.
        constr_perm = constraint_permutation_from_dec(original_model, dec_file_path)
        
        # Create an instance of ProblemPermutator with the original model.
        perm_util = ProblemPermutator(gp_env=gp_env, original_model=original_model)
        # Apply the dec permutation. (Assuming apply_dec_permutation now accepts the permutation array)
        permuted_model, _ = perm_util.apply_dec_permutation(original_model, constr_perm)
        
        # Optionally, write the permuted model to file if you need to inspect it.
        permuted_model.write(output_file)
        
        # Return the permuted model (in memory).
        return permuted_model

    except subprocess.CalledProcessError as e:
        error_msg = f"GCG failed: {e.stderr.decode().strip()}" if e.stderr else "GCG error"
        raise RuntimeError(error_msg) from e
    except Exception as e:
        raise RuntimeError(f"Processing error: {str(e)}") from e
    finally:
        # Cleanup temporary files.
        temp_files = []
        if 'cmd_file' in locals() and os.path.exists(cmd_file):
            temp_files.append(cmd_file)
        if 'temp_input' in locals() and os.path.exists(temp_input):
            temp_files.append(temp_input)
        if 'output_dir' in locals():
            for f in os.listdir(output_dir):
                # Remove any MPS or DEC files not equal to the final output file or any GCG temp files.
                if (f.endswith('.mps') and f != os.path.basename(output_file)) or f.startswith('_MPS_') or f.endswith('.dec'):
                    temp_files.append(os.path.join(output_dir, f))
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {f}: {str(e)}")


def preprocess_with_gcg(model: gp.Model, gp_env: gp.Env,
                        temp_input: str = None,
                        temp_output: str = None) -> gp.Model:
    """Enhanced workflow with unique temporary files.
    
    Writes the original model to a temporary file, runs GCG preprocessing
    (which produces a .dec file and computes a constraint permutation),
    applies that permutation to produce a new in‑memory model, cleans up all
    temporary files, and returns the permuted model.
    """
    try:
        # Generate unique temporary file names
        unique_id = uuid.uuid4().hex[:8]
        temp_input = temp_input or f"temp_permuted_{unique_id}.mps"
        temp_output = temp_output or f"temp_structured_{unique_id}.mps"
        
        temp_input = os.path.abspath(temp_input)
        temp_output = os.path.abspath(temp_output)
        
        # Write the original model to a temporary MPS file.
        model.write(temp_input)
        print(f"Original model written to: {temp_input}")
        
        # Run GCG preprocessing. This returns an in-memory permuted model.
        permuted_model = gcg_preprocess(gp_env, temp_input, temp_output)
        print("Preprocessing complete; returning in-memory permuted model.")
        
        return permuted_model

    finally:
        # Remove temporary files so that nothing is kept for analysis.
        for f in [temp_input, temp_output]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {f}: {str(e)}")
        # Additionally, remove any extra files in the output directory
        output_dir = os.path.dirname(temp_output)
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith('.mps') or f.endswith('.dec') or f.startswith('_MPS_') or f.startswith('cmds_'):
                    file_path = os.path.join(output_dir, f)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Warning: Could not remove temporary file {file_path}: {str(e)}")
