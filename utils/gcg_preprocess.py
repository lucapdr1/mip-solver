import os
import subprocess
import gurobipy as gp

import os
import subprocess
import gurobipy as gp

def gcg_preprocess(input_file: str, output_file: str = None) -> str:
    """
    Robust GCG preprocessing with visualization
    Returns path to the preprocessed file
    """
    input_file = os.path.abspath(input_file)
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_dir = f"{base}_gcg_output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"structured{ext}")
    output_file = os.path.abspath(output_file)
    output_dir = os.path.dirname(output_file)

    # Create command file
    cmd_file = os.path.join(output_dir, "gcg_commands.cmd")
    with open(cmd_file, 'w') as f:
        f.write(f'read "{input_file}"\n')
        f.write('set detection enabled TRUE\n')
        f.write('detect\n')
        f.write('explore\n')
        f.write('select 0\n')
        f.write('export 0\n')  # GCG will name the .gp file
        f.write('quit\n')
        f.write(f'write selected "{output_dir}" mps\n')
        f.write('quit\n')

    try:
        # 1. Run GCG in the output directory
        subprocess.run(
            ["gcg", "-b", cmd_file],
            cwd=output_dir,
            check=True
        )
        
        # 2. Find and process the .gp file
        gp_files = [f for f in os.listdir(output_dir) if f.endswith('.gp')]
        if gp_files:
            gp_file = os.path.join(output_dir, gp_files[0])
            vis_output = os.path.join(output_dir, "decomposition.png")
            
            # 3. Robust gnuplot conversion with error handling
            try:
                subprocess.run(
                    [
                        "gnuplot",
                        "-e",
                        f"set terminal png; set output '{vis_output}'; load '{gp_file}'"
                    ],
                    cwd=output_dir,
                    check=True,
                    capture_output=True
                )
                print(f"✓ Visualization saved to {vis_output}")
            except subprocess.CalledProcessError as e:
                print(f"Gnuplot warning: {e.stderr.decode().strip()}")
            
            # Clean up .gp file
            os.remove(gp_file)
        
        # 4. Verify output file exists
        mps_files = [f for f in os.listdir(output_dir) if f.endswith('.mps')]
        if mps_files and not os.path.exists(output_file):
            os.rename(
                os.path.join(output_dir, mps_files[0]),
                output_file
            )
            
        if not os.path.exists(output_file):
            raise FileNotFoundError("GCG failed to create output MPS file")
            
        return output_file

    except subprocess.CalledProcessError as e:
        error_msg = f"GCG processing failed: {e.stderr.decode().strip()}" if e.stderr else "GCG failed with no error output"
        raise RuntimeError(error_msg) from e
    finally:
        if os.path.exists(cmd_file):
            os.remove(cmd_file)

def preprocess_with_gcg(model: gp.Model, gp_env: gp.Env,
                       temp_input: str = "temp_permuted.mps",
                       temp_output: str = "temp_permuted_gcg_structured.mps") -> gp.Model:
    """
    Full workflow with structure preservation
    """
    temp_input = os.path.abspath(temp_input)
    temp_output = os.path.abspath(temp_output)
    
    # Write initial model
    model.write(temp_input)
    print(f"Original model written to: {temp_input}")
    
    # Run GCG preprocessing
    preprocessed_file = gcg_preprocess(temp_input, temp_output)
    
    # Load and return processed model
    processed_model = gp.read(preprocessed_file, env=gp_env)
    print(f"Preprocessed model loaded from: {preprocessed_file}")
    
    return processed_model

def visualize_structure(model_file: str):
    """Generate a matrix visualization from the structured problem"""
    viz_file = model_file.replace('.mps', '.tex')
    commands = f'''
    read "{model_file}"
    visualize matrix "{viz_file}"
    quit
    '''
    try:
        subprocess.run(
            ["gcg"],
            input=commands,
            text=True,
            shell=True,
            check=True
        )
        print(f"Visualization created: {viz_file}")
    except subprocess.CalledProcessError as e:
        print(f"Visualization failed: {e.stderr}")

def preprocess_with_gcg(model: gp.Model, gp_env: gp.Env,
                       temp_input: str = "temp_permuted.mps",
                       temp_output: str = "temp_permuted_gcg_structured.mps") -> gp.Model:
    """
    Full workflow with structure preservation
    """
    temp_input = os.path.abspath(temp_input)
    temp_output = os.path.abspath(temp_output)
    
    # Write initial model
    model.write(temp_input)
    print(f"Original model written to: {temp_input}")
    
    # Run GCG preprocessing
    preprocessed_file = gcg_preprocess(temp_input, temp_output)

    
    # Load and return processed model
    processed_model = gp.read(preprocessed_file, env=gp_env)
    print(f"Preprocessed model loaded from: {preprocessed_file}")
    
    return processed_model