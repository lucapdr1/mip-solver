resource "aws_codebuild_project" "tfer--mip-solver" {
  artifacts {
    encryption_disabled    = "false"
    override_artifact_name = "false"
    type                   = "NO_ARTIFACTS"
  }

  badge_enabled = "false"
  build_timeout = "60"

  cache {
    type = "NO_CACHE"
  }

  encryption_key = "arn:aws:kms:eu-north-1:209479294933:alias/aws/s3"

  environment {
    compute_type                = "BUILD_GENERAL1_MEDIUM"
    image                       = "aws/codebuild/amazonlinux-x86_64-standard:5.0"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode             = "false"
    type                        = "LINUX_CONTAINER"
  }

  logs_config {
    cloudwatch_logs {
      status = "ENABLED"
    }

    s3_logs {
      encryption_disabled = "false"
      status              = "DISABLED"
    }
  }

  name               = "mip-solver"
  project_visibility = "PRIVATE"
  queued_timeout     = "480"
  service_role       = "arn:aws:iam::209479294933:role/service-role/codebuild-mip-solver-service-role"

  source {
    git_clone_depth = "1"

    git_submodules_config {
      fetch_submodules = "false"
    }

    insecure_ssl        = "false"
    location            = "https://github.com/lucapdr1/mip-solver"
    report_build_status = "false"
    type                = "GITHUB"
  }

  source_version = "dockerize"
}
