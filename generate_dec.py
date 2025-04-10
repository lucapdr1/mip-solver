import os
import sys
import argparse
import concurrent.futures
from utils.dec_generator import DecGenerator
from utils.gurobi_utils import init_gurobi_env
from utils.config import NUMBER_OF_PERMUTATIONS

def process_single_file(file_path, output_dir, num_permutations):
    """
    Process a single MPS file and generate decompositions
    
    Args:
        file_path: Full path to the MPS file
        output_dir: Directory where output files will be stored
        num_permutations: Number of permutations to generate
        
    Returns:
        tuple: (file_name, success, message)
    """
    file_name = os.path.basename(file_path)
    
    try:
        # Initialize Gurobi environment (each process needs its own)
        gp_env = init_gurobi_env()
        
        # Set environment variable for output directory
        original_output_dir = os.environ.get('OUTPUT_DIR', None)
        os.environ['OUTPUT_DIR'] = output_dir
        
        # Create generator and process the file
        generator = DecGenerator(gp_env, file_path)
        generator.create_decmpositions(num_permutations)
        
        # Restore original output dir environment variable if it existed
        if original_output_dir is not None:
            os.environ['OUTPUT_DIR'] = original_output_dir
        elif 'OUTPUT_DIR' in os.environ:
            del os.environ['OUTPUT_DIR']
        
        return (file_name, True, "Success")
    except Exception as e:
        return (file_name, False, str(e))

def process_mps_files(input_dir, output_dir, num_permutations, max_workers):
    """
    Process all MPS files in the input directory in parallel
    
    Args:
        input_dir: Directory containing MPS files
        output_dir: Directory where output files will be stored
        num_permutations: Number of permutations to generate for each input
        max_workers: Maximum number of parallel processes
    """
    # Ensure directories exist
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all MPS files in the input directory
    mps_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                if f.lower().endswith('.mps')]
    
    if not mps_files:
        print(f"No MPS files found in {input_dir}")
        return
    
    print(f"Found {len(mps_files)} MPS files to process")
    print(f"Using {max_workers} parallel workers")
    
    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, file_path, output_dir, num_permutations): file_path
            for file_path in mps_files
        }
        
        # Process results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_file):
            completed += 1
            file_path = future_to_file[future]
            file_name = os.path.basename(file_path)
            
            try:
                name, success, message = future.result()
                if success:
                    print(f"[{completed}/{len(mps_files)}] ✓ Successfully processed {name}")
                else:
                    print(f"[{completed}/{len(mps_files)}] ✗ Failed to process {name}: {message}")
            except Exception as e:
                print(f"[{completed}/{len(mps_files)}] ✗ Error processing {file_name}: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process MPS files and generate decompositions")
    
    parser.add_argument("--input-dir", type=str, default="./input",
                      help="Directory containing MPS files (default: ./input)")
    parser.add_argument("--output-dir", type=str, default="./output",
                      help="Directory for output files (default: ./output)")
    parser.add_argument("--permutations", type=int, default=NUMBER_OF_PERMUTATIONS,
                      help=f"Number of permutations (default: {NUMBER_OF_PERMUTATIONS})")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                      help=f"Number of parallel workers (default: {os.cpu_count()})")
    
    args = parser.parse_args()
    
    # Use at most the number of CPUs by default, but allow override
    max_workers = min(args.workers, os.cpu_count())
    if max_workers < 1:
        max_workers = 1
    
    # Process the files
    try:
        process_mps_files(args.input_dir, args.output_dir, args.permutations, max_workers)
        print("Batch processing completed successfully")
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())