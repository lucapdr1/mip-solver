import os
import sys
import signal
import argparse
import concurrent.futures
import time
from utils.dec_generator import DecGenerator
from utils.gurobi_utils import init_gurobi_env
from utils.config import NUMBER_OF_PERMUTATIONS

"""
python generate_dec.py --input-dir="./batch_benchmark_no_dec" --output-dir="./batch_output/dec_files" --workers=2
"""

# Global flag to indicate shutdown is requested
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C by setting the global shutdown flag"""
    global shutdown_requested
    if not shutdown_requested:
        print("\nShutdown requested. Waiting for current tasks to finish...")
        shutdown_requested = True
    else:
        # If pressed twice, force exit
        print("\nForced exit.")
        sys.exit(1)

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
    global shutdown_requested
    
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
    print("Press Ctrl+C to stop processing (may wait for current tasks to complete)")
    
    # Set up pool with context manager to ensure proper cleanup
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit initial batch of tasks
        futures = {}
        for i, file_path in enumerate(mps_files):
            if shutdown_requested:
                break
            futures[executor.submit(process_single_file, file_path, output_dir, num_permutations)] = file_path
        
        # Process results as they complete
        completed = 0
        total = len(mps_files)
        
        try:
            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                file_name = os.path.basename(file_path)
                completed += 1
                
                try:
                    name, success, message = future.result()
                    if success:
                        print(f"[{completed}/{total}] ✓ Successfully processed {name}")
                    else:
                        print(f"[{completed}/{total}] ✗ Failed to process {name}: {message}")
                except Exception as e:
                    print(f"[{completed}/{total}] ✗ Error processing {file_name}: {e}")
                
                # Check if shutdown was requested
                if shutdown_requested:
                    print("Shutdown in progress. Not submitting new tasks.")
                    break
        except KeyboardInterrupt:
            print("\nShutdown requested. Cancelling pending tasks...")
            shutdown_requested = True
            
            # Cancel any pending futures
            for future in futures:
                if not future.done():
                    future.cancel()
            
            # Wait a moment for cancellation to take effect
            time.sleep(0.5)
    
    if shutdown_requested:
        print("Shutdown complete.")
    else:
        print("All files processed successfully.")

def main():
    # Set up signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    
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
        return 0
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())