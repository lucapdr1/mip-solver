resource "aws_batch_job_queue" "tfer--mip-solver-queue" {
  compute_environments = ["arn:aws:batch:eu-north-1:209479294933:compute-environment/mip-solver-on-demand-env"]
  name                 = "mip-solver-queue"
  priority             = "1"
  state                = "ENABLED"
}
