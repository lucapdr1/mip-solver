#!/bin/bash
#usage: bash ./runLocalBatchOfJobs.sh <input_dir> <output_dir> [rules_folder|parallel_instances] [parallel_instances if rules_folder provided]

# Set default values if arguments are not provided
DEFAULT_INPUT_DIR="./input/"
DEFAULT_OUTPUT_DIR="./output/"
DEFAULT_NUMBER_OF_PERMUTATIONS=3
DEFAULT_PARALLEL_INSTANCES=1

# Use provided arguments or default values
INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# Initialize variables for rules folder and parallel instances.
RULES_FOLDER=""
NUMBER_OF_PERMUTATIONS=$DEFAULT_NUMBER_OF_PERMUTATIONS
PARALLEL_INSTANCES=$DEFAULT_PARALLEL_INSTANCES

# Determine if the third parameter is a directory (rules folder) or parallel instances number.
if [ ! -z "$3" ]; then
    if [ -d "$3" ]; then
        RULES_FOLDER="$3"
        # If a fourth parameter is provided, use it as PARALLEL_INSTANCES.
        if [ ! -z "$4" ]; then
            PARALLEL_INSTANCES="$4"
        fi
    else
        PARALLEL_INSTANCES="$3"
    fi
fi

# Ensure input folder exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Input folder does not exist: $INPUT_DIR"
    exit 1
fi

# Function to process input files for a given JSON rule file
process_input_files() {
    local json_file="$1"
    local current_output_dir="$2"
    
    # Create output folder if needed
    mkdir -p "$current_output_dir"
    
    # Create a temporary file to hold file sizes and paths
    tmpfile=$(mktemp)
    
    # Collect file sizes and paths from the input folder
    find "$INPUT_DIR" -maxdepth 1 -type f -exec sh -c 'printf "%s %s\n" "$(wc -c < "$1")" "$1"' sh {} \; > "$tmpfile"
    
    # Read sorted file list into an array to preserve spaces in filenames
    mapfile -t files < <(sort -n "$tmpfile" | cut -d ' ' -f 2-)
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            INPUT_PROBLEM="$(basename "$file")"
            echo "Processing file: $INPUT_PROBLEM using rule file: $(basename "$json_file")"
            env INPUT_PROBLEM="$INPUT_PROBLEM" \
                INPUT_DIR="$INPUT_DIR" \
                OUTPUT_DIR="$current_output_dir" \
                NUMBER_OF_PERMUTATIONS="$NUMBER_OF_PERMUTATIONS" \
                python main.py "$json_file" &
            
            # If the number of background jobs equals/exceeds the limit, wait for one to finish.
            while [ "$(jobs -r | wc -l)" -ge "$PARALLEL_INSTANCES" ]; do
                sleep 0.1
            done
        fi
    done
    
    # Wait for all background processes to finish
    wait
    
    # Cleanup the temporary file
    rm "$tmpfile"
}

# Main processing:
if [ -n "$RULES_FOLDER" ]; then
    # Loop over each JSON file in the rules folder.
    for json_file in "$RULES_FOLDER"/*.json; do
        # Check if any JSON files exist
        if [ ! -e "$json_file" ]; then
            echo "No JSON files found in rules folder: $RULES_FOLDER"
            exit 1
        fi
        
        # Get the base name of the JSON file (without extension)
        rule_name=$(basename "$json_file" .json)
        # Create an output folder specific to this rule file
        current_output_dir="${OUTPUT_DIR}${rule_name}/"
        
        process_input_files "$json_file" "$current_output_dir"
    done
else
    # No rules folder provided: process input files as before.
    process_input_files "" "$OUTPUT_DIR"
fi
