import os
import argparse
import subprocess

def process_file(input_file, output_file):
    """
    Calls GCG to apply structure detection on the input MPS file and
    saves the reordered problem to the output file.
    """
    # Build the GCG command.
    # Adjust the command if your GCG installation uses different options.
    cmd = ["gcg", "-s", "detectstructure", input_file, "-o", output_file]
    
    print(f"Processing {input_file}...")
    try:
        # Run the command and capture stdout/stderr.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Processed {input_file} successfully. Output saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_file}: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(
        description="Apply GCG structure detection on a folder of MPS instances and save reordered problems."
    )
    parser.add_argument(
        "--input-folder", type=str, required=True,
        help="Folder containing MPS instances"
    )
    parser.add_argument(
        "--output-folder", type=str, required=True,
        help="Folder to save the reordered MPS instances"
    )
    args = parser.parse_args()
    
    # Ensure the output folder exists.
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # Process each MPS file in the input folder.
    for file in os.listdir(args.input_folder):
        if file.lower().endswith(".mps"):
            input_path = os.path.join(args.input_folder, file)
            base_name = os.path.splitext(file)[0]
            output_file = base_name + "_reordered.mps"
            output_path = os.path.join(args.output_folder, output_file)
            process_file(input_path, output_path)

if __name__ == "__main__":
    main()
