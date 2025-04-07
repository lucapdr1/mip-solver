resource "aws_ecr_repository" "tfer--polimi-002F-mip-solver" {
  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = "false"
  }

  image_tag_mutability = "MUTABLE"
  name                 = "polimi/mip-solver"
}
