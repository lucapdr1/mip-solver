FILENAME="dummy.mps"
JOB_NAME="mip-solver-job-1-${FILENAME%.*}"

aws batch submit-job \
  --job-name "$JOB_NAME" \
  --job-queue mip-solver-queue \
  --job-definition mip-solver-job-definition \
  --container-overrides '{
    "environment": [
      {
        "name": "INPUT_PROBLEM",
        "value": "'$FILENAME'"
      }
    ]
  }'