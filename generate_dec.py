import os
import sys
import argparse
from utils.dec_preprocess import DecGenerator
from utils.gurobi_utils import init_gurobi_env
from utils.config import NUMBER_OF_PERMUTATIONS

def process_mps_files(input_dir, output_dir, num_permutations):
    """
    Process all MPS files in the input directory and generate decompositions
    
    Args:
        input_dir: Directory containing MPS files
        output_dir: Directory where output files will be stored
        num_permutations: Number of permutations to generate for each input
    """
    # Ensure directories exist
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Gurobi environment once
    gp_env = init_gurobi_env()
    
    # Get all MPS files in the input directory
    mps_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mps')]
    
    if not mps_files:
        print(f"No MPS files found in {input_dir}")
        return
    
    print(f"Found {len(mps_files)} MPS files to process")
    
    # Process each file
    for i, file_name in enumerate(mps_files):
        file_path = os.path.join(input_dir, file_name)
        print(f"[{i+1}/{len(mps_files)}] Processing {file_name}...")
        
        try:
            # Set environment variables for the DecGenerator
            os.environ['OUTPUT_DIR'] = output_dir
            
            # Create generator and process the file
            generator = DecGenerator(gp_env, file_path)
            generator.create_decmpositions(num_permutations)
            
            print(f"✓ Successfully processed {file_name}")
        except Exception as e:
            print(f"✗ Failed to process {file_name}: {e}")
            # Continue with next file rather than terminating

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process MPS files and generate decompositions")
    
    parser.add_argument("--input-dir", type=str, default="./input",
                      help="Directory containing MPS files (default: ./input)")
    parser.add_argument("--output-dir", type=str, default="./output",
                      help="Directory for output files (default: ./output)")
    parser.add_argument("--permutations", type=int, default=NUMBER_OF_PERMUTATIONS,
                      help=f"Number of permutations (default: {NUMBER_OF_PERMUTATIONS})")
    
    args = parser.parse_args()
    
    # Process the files
    try:
        process_mps_files(args.input_dir, args.output_dir, args.permutations)
        print("Batch processing completed successfully")
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())