resource "aws_batch_compute_environment" "tfer--mip-solver-env" {
  compute_environment_name = "mip-solver-env"

  compute_resources {
    bid_percentage     = "0"
    desired_vcpus      = "0"
    max_vcpus          = "64"
    min_vcpus          = "0"
    security_group_ids = ["sg-0432b9b7d6248a8dd"]
    subnets            = ["subnet-05edb2298b8349551", "subnet-0620a73abc6be09c8", "subnet-06f93678981b5ebb9"]
    type               = "FARGATE_SPOT"
  }

  service_role = "arn:aws:iam::209479294933:role/mipRole"
  state        = "ENABLED"
  type         = "MANAGED"
}

resource "aws_batch_compute_environment" "tfer--mip-solver-on-demand-env" {
  compute_environment_name = "mip-solver-on-demand-env"

  compute_resources {
    bid_percentage     = "0"
    desired_vcpus      = "0"
    max_vcpus          = "64"
    min_vcpus          = "0"
    security_group_ids = ["sg-0432b9b7d6248a8dd"]
    subnets            = ["subnet-05edb2298b8349551", "subnet-0620a73abc6be09c8", "subnet-06f93678981b5ebb9"]
    type               = "FARGATE"
  }

  service_role = "arn:aws:iam::209479294933:role/mipRole"
  state        = "ENABLED"
  type         = "MANAGED"
}
