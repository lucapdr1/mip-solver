#!/bin/bash
#usage: sh ./runLocalBatchOfJobs.sh ./mip_lib/ ./batch_output/

# Set default values if arguments are not provided
DEFAULT_INPUT_DIR="./input/"
DEFAULT_OUTPUT_DIR="./output/"
DEFAULT_NUMBER_OF_PERMUTATIONS=4

# Use provided arguments or default values
INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# Initialize variables for rules folder and number of permutations.
RULES_FOLDER=""
NUMBER_OF_PERMUTATIONS=$DEFAULT_NUMBER_OF_PERMUTATIONS

# Determine if the third parameter is a directory (rules folder) or a permutation number.
if [ ! -z "$3" ]; then
    if [ -d "$3" ]; then
        RULES_FOLDER="$3"
        # If a fourth parameter is provided, use it as NUMBER_OF_PERMUTATIONS.
        if [ ! -z "$4" ]; then
            NUMBER_OF_PERMUTATIONS="$4"
        fi
    else
        NUMBER_OF_PERMUTATIONS="$3"
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
    
    # Process files in order of increasing size
    sort -n "$tmpfile" | cut -d ' ' -f 2- | while IFS= read -r file; do
        if [ -f "$file" ]; then
            INPUT_PROBLEM="$(basename "$file")"
            echo "Processing file: $INPUT_PROBLEM using rule file: $(basename "$json_file")"
            env INPUT_PROBLEM="$INPUT_PROBLEM" \
                INPUT_DIR="$INPUT_DIR" \
                OUTPUT_DIR="$current_output_dir" \
                NUMBER_OF_PERMUTATIONS="$NUMBER_OF_PERMUTATIONS" \
                python main.py "$json_file"
        fi
    done
    
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
