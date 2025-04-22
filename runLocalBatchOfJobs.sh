#!/bin/bash
# Usage:
#   bash ./runLocalBatchOfJobs.sh [--input-dir=<dir>] [--output-dir=<dir>] [--rules-folder=<dir>]
#                                  [--parallel-instances=<num>] [--number-of-permutations=<num>]
#                                  [--permute-granularity=<value>] [--time-limit=<value>] [--threads=<num>]
#
# Example (using GNU parallel):
#   parallel ./runLocalBatchOfJobs.sh --input-dir=./batch_easy/ --output-dir=./batch_output/granularity_{} \
#           --parallel-instances=2 --permute-granularity={} --time-limit=3600 --threads=8 ::: 5 6 8 10 12 15 20 33 all
#
# Enable job control
set -m

trap 'echo "Interrupted. Killing all child processes..."; pkill -P $$; exit 1' SIGINT SIGTERM

# Default values:
DEFAULT_INPUT_DIR="./input/"
DEFAULT_OUTPUT_DIR="./output/"
DEFAULT_NUMBER_OF_PERMUTATIONS=3
DEFAULT_PARALLEL_INSTANCES=1
DEFAULT_PERMUTE_GRANULARITY_K="all"
DEFAULT_TIME_LIMIT=3600
DEFAULT_THREADS=8

print_help() {
    echo "Usage: bash ./runLocalBatchOfJobs.sh [--input-dir=<dir>] [--output-dir=<dir>] [--rules-folder=<dir>]"
    echo "                                  [--parallel-instances=<num>] [--number-of-permutations=<num>]"
    echo "                                  [--permute-granularity=<value>] [--time-limit=<value>] [--threads=<num>]"
    echo ""
    echo "Default values:"
    echo "  INPUT_DIR: ${DEFAULT_INPUT_DIR}"
    echo "  OUTPUT_DIR: ${DEFAULT_OUTPUT_DIR}"
    echo "  NUMBER_OF_PERMUTATIONS: ${DEFAULT_NUMBER_OF_PERMUTATIONS}"
    echo "  PARALLEL_INSTANCES: ${DEFAULT_PARALLEL_INSTANCES}"
    echo "  PERMUTE_GRANULARITY_K: ${DEFAULT_PERMUTE_GRANULARITY_K}"
    echo "  TIME_LIMIT: ${DEFAULT_TIME_LIMIT}"
    echo "  NUMBER_OF_THREADS: ${DEFAULT_THREADS}"
}

# Check if help is requested
for arg in "$@"; do
    case $arg in
        --help|-h)
            print_help
            exit 0
            ;;
    esac
done

# Initialize variables with default values
INPUT_DIR="$DEFAULT_INPUT_DIR"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
RULES_FOLDER=""
NUMBER_OF_PERMUTATIONS="$DEFAULT_NUMBER_OF_PERMUTATIONS"
PARALLEL_INSTANCES="$DEFAULT_PARALLEL_INSTANCES"
PERMUTE_GRANULARITY_K="$DEFAULT_PERMUTE_GRANULARITY_K"
TIME_LIMIT="$DEFAULT_TIME_LIMIT"
NUMBER_OF_THREADS="$DEFAULT_THREADS"

# Parse named parameters
for arg in "$@"; do
    case $arg in
        --input-dir=*) INPUT_DIR="${arg#*=}" ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
        --rules-folder=*) RULES_FOLDER="${arg#*=}" ;;
        --parallel-instances=*) PARALLEL_INSTANCES="${arg#*=}" ;;
        --number-of-permutations=*) NUMBER_OF_PERMUTATIONS="${arg#*=}" ;;
        --permute-granularity=*) PERMUTE_GRANULARITY_K="${arg#*=}" ;;
        --time-limit=*) TIME_LIMIT="${arg#*=}" ;;
        --threads=*) NUMBER_OF_THREADS="${arg#*=}" ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# Ensure input folder exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Input folder does not exist: $INPUT_DIR"
    exit 1
fi

process_input_files() {
    local json_file="$1"
    local current_output_dir="$2"

    mkdir -p "$current_output_dir"
    tmpfile=$(mktemp)

    # List input .mps files sorted by size
    find "$INPUT_DIR" -maxdepth 1 -type f -name "*.mps" -exec sh -c 'printf "%s %s\n" "$(wc -c < "$1")" "$1"' sh {} \; > "$tmpfile"
    mapfile -t files < <(sort -n "$tmpfile" | cut -d ' ' -f 2-)

    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            INPUT_PROBLEM="$(basename "$file")"
            echo "Processing: $INPUT_PROBLEM with $(basename "$json_file")"

            env INPUT_PROBLEM="$INPUT_PROBLEM" \
                INPUT_DIR="$INPUT_DIR" \
                OUTPUT_DIR="$current_output_dir" \
                NUMBER_OF_PERMUTATIONS="$NUMBER_OF_PERMUTATIONS" \
                PERMUTE_GRANULARITY_K="$PERMUTE_GRANULARITY_K" \
                MAX_SOLVE_TIME="$TIME_LIMIT" \
                NUMBER_OF_THREADS="$NUMBER_OF_THREADS" \
                python main.py "$json_file" &

            # Throttle concurrent jobs
            if [ "$(jobs -p | wc -l)" -ge "$PARALLEL_INSTANCES" ]; then
                wait -n
            fi
        fi
    done

    # Wait for any remaining jobs to finish
    wait

    rm "$tmpfile"
}

# Main loop
if [ -n "$RULES_FOLDER" ]; then
    for json_file in "$RULES_FOLDER"/*.json; do
        if [ ! -e "$json_file" ]; then
            echo "No JSON rule files found in: $RULES_FOLDER"
            exit 1
        fi

        rule_name=$(basename "$json_file" .json)
        current_output_dir="${OUTPUT_DIR}${rule_name}/"

        process_input_files "$json_file" "$current_output_dir"
    done
else
    process_input_files "" "$OUTPUT_DIR"
fi
