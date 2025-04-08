import os
import re
import shutil
import argparse

def main(logs_dir, mps_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Regex to match log filenames ending in .mps.log or .mps.gz.log
    pattern = re.compile(r".*_(.+?\.mps)(?:\.gz)?\.log$")

    copied = set()
    for root, _, files in os.walk(logs_dir):
        for file in files:
            match = pattern.match(file)
            if match:
                mps_file = match.group(1)
                if mps_file not in copied:
                    source_path = os.path.join(mps_dir, mps_file)
                    if os.path.exists(source_path):
                        shutil.copy(source_path, os.path.join(output_dir, mps_file))
                        copied.add(mps_file)
                    else:
                        print(f"⚠️  Missing: {mps_file} not found in {mps_dir}")

    print(f"\n✅ Copied {len(copied)} MPS files to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MPS subset based on log files.")
    parser.add_argument("--logs-dir", required=True, help="Path to the logs base directory")
    parser.add_argument("--mps-dir", required=True, help="Path to the directory containing all .mps files")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory for subset .mps files")

    args = parser.parse_args()
    main(args.logs_dir, args.mps_dir, args.output_dir)
