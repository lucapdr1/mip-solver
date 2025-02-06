#!/bin/bash

# Usage: ./submit_jobs.sh <input_directory> <num_permutations>

INPUT_DIR="$1"
NUM_PERMUTATIONS="${2:-}"  # Default to empty if not provided

# Verify input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory $INPUT_DIR does not exist"
  exit 1
fi

# Counter for job numbering
JOB_COUNT=1

# Process each file in input directory
for file in "$INPUT_DIR"/*.mps; do
  # Get just the filename without path
  FILENAME=$(basename "$file")
  
  # Create unique job name
  JOB_NAME="mip-solver-job-${JOB_COUNT}-${FILENAME%.*}"

  # Base command
  CMD="aws batch submit-job \
    --job-name \"$JOB_NAME\" \
    --job-queue mip-solver-queue \
    --job-definition mip-solver-job-definition \
    --container-overrides '{
      \"environment\": ["

  # Add NUMBER_OF_PERMUATATIONS if provided
  if [ -n "$NUM_PERMUTATIONS" ]; then
    CMD+="
        {
          \"name\": \"NUMBER_OF_PERMUATATIONS\",
          \"value\": \"$NUM_PERMUTATIONS\"
        },"
  fi

  # Add INPUT_PROBLEM and close JSON
  CMD+="
        {
          \"name\": \"INPUT_PROBLEM\",
          \"value\": \"$FILENAME\"
        }
      ]
    }'"

  # Execute command
  echo "Submitting job for $FILENAME..."
  eval "$CMD"
  
  # Increment counter
  ((JOB_COUNT++))
done

echo "Submitted $((JOB_COUNT - 1)) jobs"