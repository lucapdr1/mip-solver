#!/bin/bash
# Usage:
#   bash ./runLocalBatchOfJobs.sh [--input-dir=<dir>] [--output-dir=<dir>] [--rules-folder=<dir>]
#                                  [--parallel-instances=<num>] [--number-of-permutations=<num>]
#                                  [--permute-granularity=<value>] [--time-limit=<value>]
#
# Example (using GNU parallel):
#   parallel ./runLocalBatchOfJobs.sh --input-dir=./batch_easy/ --output-dir=./batch_output/granularity_{} \
#           --parallel-instances=2 --permute-granularity={} --time-limit=3600 ::: 5 6 8 10 12 15 20 33 all
#
# Default values:
DEFAULT_INPUT_DIR="./input/"
DEFAULT_OUTPUT_DIR="./output/"
DEFAULT_NUMBER_OF_PERMUTATIONS=3
DEFAULT_PARALLEL_INSTANCES=1
DEFAULT_PERMUTE_GRANULARITY_K="all"
DEFAULT_TIME_LIMIT=3600

# Function to display help message.
print_help() {
    echo "Usage: bash ./runLocalBatchOfJobs.sh [--input-dir=<dir>] [--output-dir=<dir>] [--rules-folder=<dir>]"
    echo "                                  [--parallel-instances=<num>] [--number-of-permutations=<num>]"
    echo "                                  [--permute-granularity=<value>] [--time-limit=<value>]"
    echo ""
    echo "Default values:"
    echo "  INPUT_DIR: ${DEFAULT_INPUT_DIR}"
    echo "  OUTPUT_DIR: ${DEFAULT_OUTPUT_DIR}"
    echo "  NUMBER_OF_PERMUTATIONS: ${DEFAULT_NUMBER_OF_PERMUTATIONS}"
    echo "  PARALLEL_INSTANCES: ${DEFAULT_PARALLEL_INSTANCES}"
    echo "  PERMUTE_GRANULARITY_K: ${DEFAULT_PERMUTE_GRANULARITY_K}"
    echo "  TIME_LIMIT: ${DEFAULT_TIME_LIMIT}"
    echo ""
    echo "Example (using GNU parallel):"
    echo "  parallel ./runLocalBatchOfJobs.sh --input-dir=./batch_easy/ \\"
    echo "         --output-dir=./batch_output/granularity_{} --parallel-instances=2 \\"
    echo "         --permute-granularity={} --time-limit=3600 ::: 5 6 8 10 12 15 20 33 all"
}

# Check if help is requested.
for arg in "$@"; do
    case $arg in
        --help|-h)
            print_help
            exit 0
            ;;
    esac
done

# Initialize variables with default values.
INPUT_DIR="$DEFAULT_INPUT_DIR"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
RULES_FOLDER=""
NUMBER_OF_PERMUTATIONS="$DEFAULT_NUMBER_OF_PERMUTATIONS"
PARALLEL_INSTANCES="$DEFAULT_PARALLEL_INSTANCES"
PERMUTE_GRANULARITY_K="$DEFAULT_PERMUTE_GRANULARITY_K"
TIME_LIMIT="$DEFAULT_TIME_LIMIT"

# Parse named parameters.
for arg in "$@"; do
    case $arg in
        --input-dir=*)
            INPUT_DIR="${arg#*=}"
            shift
            ;;
        --output-dir=*)
            OUTPUT_DIR="${arg#*=}"
            shift
            ;;
        --rules-folder=*)
            RULES_FOLDER="${arg#*=}"
            shift
            ;;
        --parallel-instances=*)
            PARALLEL_INSTANCES="${arg#*=}"
            shift
            ;;
        --number-of-permutations=*)
            NUMBER_OF_PERMUTATIONS="${arg#*=}"
            shift
            ;;
        --permute-granularity=*)
            PERMUTE_GRANULARITY_K="${arg#*=}"
            shift
            ;;
        --time-limit=*)
            TIME_LIMIT="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Ensure input folder exists.
if [ ! -d "$INPUT_DIR" ]; then
    echo "Input folder does not exist: $INPUT_DIR"
    exit 1
fi

# Function to process input files for a given JSON rule file.
process_input_files() {
    local json_file="$1"
    local current_output_dir="$2"
    
    # Create output folder if needed.
    mkdir -p "$current_output_dir"
    
    # Create a temporary file to hold file sizes and paths.
    tmpfile=$(mktemp)
    
    # Collect file sizes and paths from the input folder.
    find "$INPUT_DIR" -maxdepth 1 -type f -exec sh -c 'printf "%s %s\n" "$(wc -c < "$1")" "$1"' sh {} \; > "$tmpfile"
    
    # Read sorted file list into an array (preserving spaces in filenames).
    mapfile -t files < <(sort -n "$tmpfile" | cut -d ' ' -f 2-)
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            INPUT_PROBLEM="$(basename "$file")"
            echo "Processing file: $INPUT_PROBLEM using rule file: $(basename "$json_file")"
            # Pass all necessary parameters as environment variables.
            env INPUT_PROBLEM="$INPUT_PROBLEM" \
                INPUT_DIR="$INPUT_DIR" \
                OUTPUT_DIR="$current_output_dir" \
                NUMBER_OF_PERMUTATIONS="$NUMBER_OF_PERMUTATIONS" \
                PERMUTE_GRANULARITY_K="$PERMUTE_GRANULARITY_K" \
                MAX_SOLVE_TIME="$TIME_LIMIT" \
                python main.py "$json_file" &
            
            # Wait if the number of background jobs equals or exceeds the parallel instances limit.
            while [ "$(jobs -r | wc -l)" -ge "$PARALLEL_INSTANCES" ]; do
                sleep 0.1
            done
        fi
    done
    
    # Wait for all background processes to finish.
    wait
    
    # Cleanup the temporary file.
    rm "$tmpfile"
}

# Main processing:
if [ -n "$RULES_FOLDER" ]; then
    # Loop over each JSON file in the rules folder.
    for json_file in "$RULES_FOLDER"/*.json; do
        # Check if any JSON files exist.
        if [ ! -e "$json_file" ]; then
            echo "No JSON files found in rules folder: $RULES_FOLDER"
            exit 1
        fi
        
        # Get the base name of the JSON file (without extension).
        rule_name=$(basename "$json_file" .json)
        # Create an output folder specific to this rule file.
        current_output_dir="${OUTPUT_DIR}${rule_name}/"
        
        process_input_files "$json_file" "$current_output_dir"
    done
else
    # No rules folder provided: process input files using the default or specified output folder.
    process_input_files "" "$OUTPUT_DIR"
fi
