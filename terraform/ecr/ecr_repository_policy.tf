resource "aws_ecr_repository_policy" "tfer--polimi-002F-mip-solver" {
  policy = <<POLICY
{
  "Statement": [
    {
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability"
      ],
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::209479294933:role/aws-service-role/batch.amazonaws.com/AWSServiceRoleForBatch"
      }
    }
  ],
  "Version": "2012-10-17"
}
POLICY

  repository = "polimi/mip-solver"
}
