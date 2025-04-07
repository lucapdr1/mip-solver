resource "aws_iam_role" "tfer--AWSServiceRoleForBatch" {
  assume_role_policy = <<POLICY
{
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Effect": "Allow",
      "Principal": {
        "Service": "batch.amazonaws.com"
      }
    }
  ],
  "Version": "2012-10-17"
}
POLICY

  managed_policy_arns  = ["arn:aws:iam::aws:policy/aws-service-role/BatchServiceRolePolicy"]
  max_session_duration = "3600"
  name                 = "AWSServiceRoleForBatch"
  path                 = "/aws-service-role/batch.amazonaws.com/"
}

resource "aws_iam_role" "tfer--AWSServiceRoleForECS" {
  assume_role_policy = <<POLICY
{
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs.amazonaws.com"
      }
    }
  ],
  "Version": "2012-10-17"
}
POLICY

  description          = "Policy to enable Amazon ECS to manage your EC2 instances and related resources."
  managed_policy_arns  = ["arn:aws:iam::aws:policy/aws-service-role/AmazonECSServiceRolePolicy"]
  max_session_duration = "3600"
  name                 = "AWSServiceRoleForECS"
  path                 = "/aws-service-role/ecs.amazonaws.com/"
}

resource "aws_iam_role" "tfer--AWSServiceRoleForSupport" {
  assume_role_policy = <<POLICY
{
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Effect": "Allow",
      "Principal": {
        "Service": "support.amazonaws.com"
      }
    }
  ],
  "Version": "2012-10-17"
}
POLICY

  description          = "Enables resource access for AWS to provide billing, administrative and support services"
  managed_policy_arns  = ["arn:aws:iam::aws:policy/aws-service-role/AWSSupportServiceRolePolicy"]
  max_session_duration = "3600"
  name                 = "AWSServiceRoleForSupport"
  path                 = "/aws-service-role/support.amazonaws.com/"
}

resource "aws_iam_role" "tfer--AWSServiceRoleForTrustedAdvisor" {
  assume_role_policy = <<POLICY
{
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Effect": "Allow",
      "Principal": {
        "Service": "trustedadvisor.amazonaws.com"
      }
    }
  ],
  "Version": "2012-10-17"
}
POLICY

  description          = "Access for the AWS Trusted Advisor Service to help reduce cost, increase performance, and improve security of your AWS environment."
  managed_policy_arns  = ["arn:aws:iam::aws:policy/aws-service-role/AWSTrustedAdvisorServiceRolePolicy"]
  max_session_duration = "3600"
  name                 = "AWSServiceRoleForTrustedAdvisor"
  path                 = "/aws-service-role/trustedadvisor.amazonaws.com/"
}

resource "aws_iam_role" "tfer--codebuild-mip-solver-service-role" {
  assume_role_policy = <<POLICY
{
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Effect": "Allow",
      "Principal": {
        "Service": "codebuild.amazonaws.com"
      }
    }
  ],
  "Version": "2012-10-17"
}
POLICY

  managed_policy_arns  = ["arn:aws:iam::209479294933:policy/service-role/CodeBuildBasePolicy-mip-solver-eu-north-1", "arn:aws:iam::209479294933:policy/service-role/CodeBuildCodeConnectionsSourceCredentialsPolicy-mip-solver-eu-north-1", "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"]
  max_session_duration = "3600"
  name                 = "codebuild-mip-solver-service-role"
  path                 = "/service-role/"
}

resource "aws_iam_role" "tfer--mipRole" {
  assume_role_policy = <<POLICY
{
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::209479294933:user/luca",
        "Service": [
          "ecs-tasks.amazonaws.com",
          "batch.amazonaws.com"
        ]
      },
      "Sid": "Statement1"
    }
  ],
  "Version": "2012-10-17"
}
POLICY

  managed_policy_arns  = ["arn:aws:iam::209479294933:policy/mipServicePolicy", "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPullOnly", "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly", "arn:aws:iam::aws:policy/SecretsManagerReadWrite", "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole", "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"]
  max_session_duration = "3600"
  name                 = "mipRole"
  path                 = "/"
}
