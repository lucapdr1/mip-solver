resource "aws_iam_role_policy_attachment" "tfer--AWSServiceRoleForBatch_BatchServiceRolePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/aws-service-role/BatchServiceRolePolicy"
  role       = "AWSServiceRoleForBatch"
}

resource "aws_iam_role_policy_attachment" "tfer--AWSServiceRoleForECS_AmazonECSServiceRolePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/aws-service-role/AmazonECSServiceRolePolicy"
  role       = "AWSServiceRoleForECS"
}

resource "aws_iam_role_policy_attachment" "tfer--AWSServiceRoleForSupport_AWSSupportServiceRolePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/aws-service-role/AWSSupportServiceRolePolicy"
  role       = "AWSServiceRoleForSupport"
}

resource "aws_iam_role_policy_attachment" "tfer--AWSServiceRoleForTrustedAdvisor_AWSTrustedAdvisorServiceRolePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/aws-service-role/AWSTrustedAdvisorServiceRolePolicy"
  role       = "AWSServiceRoleForTrustedAdvisor"
}

resource "aws_iam_role_policy_attachment" "tfer--codebuild-mip-solver-service-role_AmazonEC2ContainerRegistryPowerUser" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
  role       = "codebuild-mip-solver-service-role"
}

resource "aws_iam_role_policy_attachment" "tfer--codebuild-mip-solver-service-role_CodeBuildBasePolicy-mip-solver-eu-north-1" {
  policy_arn = "arn:aws:iam::209479294933:policy/service-role/CodeBuildBasePolicy-mip-solver-eu-north-1"
  role       = "codebuild-mip-solver-service-role"
}

resource "aws_iam_role_policy_attachment" "tfer--codebuild-mip-solver-service-role_CodeBuildCodeConnectionsSourceCredentialsPolicy-mip-solver-eu-north-1" {
  policy_arn = "arn:aws:iam::209479294933:policy/service-role/CodeBuildCodeConnectionsSourceCredentialsPolicy-mip-solver-eu-north-1"
  role       = "codebuild-mip-solver-service-role"
}

resource "aws_iam_role_policy_attachment" "tfer--mipRole_AWSBatchServiceRole" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
  role       = "mipRole"
}

resource "aws_iam_role_policy_attachment" "tfer--mipRole_AmazonEC2ContainerRegistryPullOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPullOnly"
  role       = "mipRole"
}

resource "aws_iam_role_policy_attachment" "tfer--mipRole_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = "mipRole"
}

resource "aws_iam_role_policy_attachment" "tfer--mipRole_AmazonECSTaskExecutionRolePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
  role       = "mipRole"
}

resource "aws_iam_role_policy_attachment" "tfer--mipRole_SecretsManagerReadWrite" {
  policy_arn = "arn:aws:iam::aws:policy/SecretsManagerReadWrite"
  role       = "mipRole"
}

resource "aws_iam_role_policy_attachment" "tfer--mipRole_mipServicePolicy" {
  policy_arn = "arn:aws:iam::209479294933:policy/mipServicePolicy"
  role       = "mipRole"
}
