#!/bin/bash

# Usage example sh ./runBatchOfJobs.sh ./batch experiments2/
# To download experiment result
# aws s3 cp s3://lucapolimi-experiments/firstBounds/ ./batch_output/frist-bounds --recursive

INPUT_DIR="$1"
OUTPUT_DIR="$2"
NUM_PERMUTATIONS="${3:5}"  # Default to 5 if not provided

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

  # Add NUMBER_OF_PERMUTATIONS if provided
  if [ -n "$NUM_PERMUTATIONS" ]; then
    CMD+="
        {
          \"name\": \"NUMBER_OF_PERMUTATIONS\",
          \"value\": \"$NUM_PERMUTATIONS\"
        },"
  fi

  # Add INPUT_PROBLEM and close JSON
  CMD+="
        {
          \"name\": \"INPUT_PROBLEM\",
          \"value\": \"$FILENAME\"
        },
         "

  CMD+="
        {
          \"name\": \"OUTPUT_DIR\",
          \"value\": \"$OUTPUT_DIR\"
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