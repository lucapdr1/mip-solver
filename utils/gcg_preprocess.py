import os
import shutil
import subprocess
import uuid
import gurobipy as gp
from datetime import datetime

def process_gp_file(output_dir: str, base_name: str) -> None:
    """Process .gp file and handle GCG's auto-generated PDF output."""
    try:
        # 1. Find and rename the GCG-generated .gp file
        gp_files = [f for f in os.listdir(output_dir) 
                   if f.endswith('.gp') and f.startswith('_MPS_')]
        
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

def gcg_preprocess(input_file: str, output_file: str = None) -> str:
    """Robust GCG preprocessing with strict file handling."""
    # Generate unique ID upfront
    unique_id = uuid.uuid4().hex[:8]
    
    try:
        input_file = os.path.abspath(input_file)
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_dir = f"{base}_gcg_output_{unique_id}"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"structured_{unique_id}{ext}")
        else:
            output_file = os.path.abspath(output_file)
            output_dir = os.path.dirname(output_file)
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

        # Create a temporary copy of input file in output directory
        temp_input = os.path.join(output_dir, f"temp_input_{unique_id}.mps")
        shutil.copy2(input_file, temp_input)

        # Unique command file name
        cmd_file = os.path.join(output_dir, f"cmds_{unique_id}.cmd")
        with open(cmd_file, 'w') as f:
            f.write(f'read "{temp_input}"\n')
            f.write('set detection enabled TRUE\n')
            f.write('detect\n')
            f.write('explore\n')
            f.write('select 0\n')
            f.write('export 0\n')
            f.write('quit\n')
            f.write(f'write selected "{output_dir}" mps\n')
            f.write('quit\n')

        # Run GCG with timeout
        subprocess.run(
            ["gcg", "-b", cmd_file],
            cwd=output_dir,
            check=True,
            timeout=300  # 5 minutes timeout
        )
        
        # Process visualization
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        process_gp_file(output_dir, base_name)

        # Handle output file naming - look for newest MPS that's not our temp input
        mps_files = []
        for f in os.listdir(output_dir):
            if f.endswith('.mps') and f != os.path.basename(temp_input):
                mps_files.append((f, os.path.getctime(os.path.join(output_dir, f))))
        
        if mps_files:
            # Get most recently created MPS file
            mps_files.sort(key=lambda x: x[1], reverse=True)
            latest_mps = os.path.join(output_dir, mps_files[0][0])
            
            # Ensure we don't overwrite existing output
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rename(latest_mps, output_file)
        
        if not os.path.exists(output_file):
            raise FileNotFoundError("GCG failed to create output MPS file")
        
        return output_file

    except subprocess.CalledProcessError as e:
        error_msg = f"GCG failed: {e.stderr.decode().strip()}" if e.stderr else "GCG error"
        raise RuntimeError(error_msg) from e
    except Exception as e:
        raise RuntimeError(f"Processing error: {str(e)}") from e
    finally:
        # Cleanup temporary files
        temp_files = []
        if 'cmd_file' in locals() and os.path.exists(cmd_file):
            temp_files.append(cmd_file)
        if 'temp_input' in locals() and os.path.exists(temp_input):
            temp_files.append(temp_input)
        if 'output_dir' in locals():
            for f in os.listdir(output_dir):
                if (f.endswith('.mps') and f != os.path.basename(output_file)) or \
                   f.startswith('_MPS_'):
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
    """Enhanced workflow with unique temporary files."""
    try:
        # Generate unique temp names
        unique_id = uuid.uuid4().hex[:8]
        temp_input = temp_input or f"temp_permuted_{unique_id}.mps"
        temp_output = temp_output or f"temp_structured_{unique_id}.mps"
        
        temp_input = os.path.abspath(temp_input)
        temp_output = os.path.abspath(temp_output)
        
        # Write initial model
        model.write(temp_input)
        print(f"Original model written to: {temp_input}")
        
        # Run preprocessing
        preprocessed_file = gcg_preprocess(temp_input, temp_output)
        
        # Load model
        processed_model = gp.read(preprocessed_file, env=gp_env)
        print(f"Preprocessed model loaded from: {preprocessed_file}")
        
        return processed_model
    finally:
        # Cleanup temporary files
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)
        # Keep final output for analysis