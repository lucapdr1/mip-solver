output "aws_iam_policy_tfer--CodeBuildBasePolicy-mip-solver-eu-north-1_id" {
  value = "${aws_iam_policy.tfer--CodeBuildBasePolicy-mip-solver-eu-north-1.id}"
}

output "aws_iam_policy_tfer--CodeBuildCodeConnectionsSourceCredentialsPolicy-mip-solver-eu-north-1_id" {
  value = "${aws_iam_policy.tfer--CodeBuildCodeConnectionsSourceCredentialsPolicy-mip-solver-eu-north-1.id}"
}

output "aws_iam_policy_tfer--mipServicePolicy_id" {
  value = "${aws_iam_policy.tfer--mipServicePolicy.id}"
}

output "aws_iam_role_policy_attachment_tfer--AWSServiceRoleForBatch_BatchServiceRolePolicy_id" {
  value = "${aws_iam_role_policy_attachment.tfer--AWSServiceRoleForBatch_BatchServiceRolePolicy.id}"
}

output "aws_iam_role_policy_attachment_tfer--AWSServiceRoleForECS_AmazonECSServiceRolePolicy_id" {
  value = "${aws_iam_role_policy_attachment.tfer--AWSServiceRoleForECS_AmazonECSServiceRolePolicy.id}"
}

output "aws_iam_role_policy_attachment_tfer--AWSServiceRoleForSupport_AWSSupportServiceRolePolicy_id" {
  value = "${aws_iam_role_policy_attachment.tfer--AWSServiceRoleForSupport_AWSSupportServiceRolePolicy.id}"
}

output "aws_iam_role_policy_attachment_tfer--AWSServiceRoleForTrustedAdvisor_AWSTrustedAdvisorServiceRolePolicy_id" {
  value = "${aws_iam_role_policy_attachment.tfer--AWSServiceRoleForTrustedAdvisor_AWSTrustedAdvisorServiceRolePolicy.id}"
}

output "aws_iam_role_policy_attachment_tfer--codebuild-mip-solver-service-role_AmazonEC2ContainerRegistryPowerUser_id" {
  value = "${aws_iam_role_policy_attachment.tfer--codebuild-mip-solver-service-role_AmazonEC2ContainerRegistryPowerUser.id}"
}

output "aws_iam_role_policy_attachment_tfer--codebuild-mip-solver-service-role_CodeBuildBasePolicy-mip-solver-eu-north-1_id" {
  value = "${aws_iam_role_policy_attachment.tfer--codebuild-mip-solver-service-role_CodeBuildBasePolicy-mip-solver-eu-north-1.id}"
}

output "aws_iam_role_policy_attachment_tfer--codebuild-mip-solver-service-role_CodeBuildCodeConnectionsSourceCredentialsPolicy-mip-solver-eu-north-1_id" {
  value = "${aws_iam_role_policy_attachment.tfer--codebuild-mip-solver-service-role_CodeBuildCodeConnectionsSourceCredentialsPolicy-mip-solver-eu-north-1.id}"
}

output "aws_iam_role_policy_attachment_tfer--mipRole_AWSBatchServiceRole_id" {
  value = "${aws_iam_role_policy_attachment.tfer--mipRole_AWSBatchServiceRole.id}"
}

output "aws_iam_role_policy_attachment_tfer--mipRole_AmazonEC2ContainerRegistryPullOnly_id" {
  value = "${aws_iam_role_policy_attachment.tfer--mipRole_AmazonEC2ContainerRegistryPullOnly.id}"
}

output "aws_iam_role_policy_attachment_tfer--mipRole_AmazonEC2ContainerRegistryReadOnly_id" {
  value = "${aws_iam_role_policy_attachment.tfer--mipRole_AmazonEC2ContainerRegistryReadOnly.id}"
}

output "aws_iam_role_policy_attachment_tfer--mipRole_AmazonECSTaskExecutionRolePolicy_id" {
  value = "${aws_iam_role_policy_attachment.tfer--mipRole_AmazonECSTaskExecutionRolePolicy.id}"
}

output "aws_iam_role_policy_attachment_tfer--mipRole_SecretsManagerReadWrite_id" {
  value = "${aws_iam_role_policy_attachment.tfer--mipRole_SecretsManagerReadWrite.id}"
}

output "aws_iam_role_policy_attachment_tfer--mipRole_mipServicePolicy_id" {
  value = "${aws_iam_role_policy_attachment.tfer--mipRole_mipServicePolicy.id}"
}

output "aws_iam_role_tfer--AWSServiceRoleForBatch_id" {
  value = "${aws_iam_role.tfer--AWSServiceRoleForBatch.id}"
}

output "aws_iam_role_tfer--AWSServiceRoleForECS_id" {
  value = "${aws_iam_role.tfer--AWSServiceRoleForECS.id}"
}

output "aws_iam_role_tfer--AWSServiceRoleForSupport_id" {
  value = "${aws_iam_role.tfer--AWSServiceRoleForSupport.id}"
}

output "aws_iam_role_tfer--AWSServiceRoleForTrustedAdvisor_id" {
  value = "${aws_iam_role.tfer--AWSServiceRoleForTrustedAdvisor.id}"
}

output "aws_iam_role_tfer--codebuild-mip-solver-service-role_id" {
  value = "${aws_iam_role.tfer--codebuild-mip-solver-service-role.id}"
}

output "aws_iam_role_tfer--mipRole_id" {
  value = "${aws_iam_role.tfer--mipRole.id}"
}
