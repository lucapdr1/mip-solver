#!/bin/bash

# Set default values if arguments are not provided
DEFAULT_INPUT_DIR="./input"
DEFAULT_OUTPUT_DIR="./output"
DEFAULT_NUMBER_OF_PERMUTATIONS=5

# Use provided arguments or default values
INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}/"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}/"
NUMBER_OF_PERMUTATIONS="${3:-$DEFAULT_NUMBER_OF_PERMUTATIONS}"

# Ensure input folder exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Input folder does not exist: $INPUT_DIR"
    exit 1
fi

# Run the Python script for every file in the input folder
for file in "$INPUT_DIR"/*; do
    if [ -f "$file" ]; then
        INPUT_PROBLEM="$(basename "$file")"
        echo "Processing file: $INPUT_PROBLEM"
        env INPUT_PROBLEM="$INPUT_PROBLEM" INPUT_DIR="$INPUT_DIR" OUTPUT_DIR="$OUTPUT_DIR" NUMBER_OF_PERMUTATIONS="$NUMBER_OF_PERMUTATIONS" python main.py
    fi
done
