resource "aws_ecs_cluster" "tfer--mip-solver-env_Batch_6fe563a1-fbcf-3544-a5c4-ec546712c2ba" {
  name = "mip-solver-env_Batch_6fe563a1-fbcf-3544-a5c4-ec546712c2ba"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_cluster" "tfer--mip-solver-on-demand-env_Batch_23239dbf-056c-3cfa-96e7-4ea7d65d1e36" {
  name = "mip-solver-on-demand-env_Batch_23239dbf-056c-3cfa-96e7-4ea7d65d1e36"

  setting {
    name  = "containerInsights"
    value = "disabled"
  }
}
