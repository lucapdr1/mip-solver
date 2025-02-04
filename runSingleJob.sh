aws batch submit-job \
  --job-name "mip-solver-job-1" \
  --job-queue mip-solver-queue \
  --job-definition mip-solver-job-definition \
  --container-overrides '{
    "environment": [
      {
        "name": "INPUT_PROBLEM",
        "value": "input/dummy.mps"
      }
    ]
  }'