resource "aws_iam_policy" "tfer--CodeBuildBasePolicy-mip-solver-eu-north-1" {
  description = "Policy used in trust relationship with CodeBuild"
  name        = "CodeBuildBasePolicy-mip-solver-eu-north-1"
  path        = "/service-role/"

  policy = <<POLICY
{
  "Statement": [
    {
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Effect": "Allow",
      "Resource": [
        "arn:aws:logs:eu-north-1:209479294933:log-group:/aws/codebuild/mip-solver",
        "arn:aws:logs:eu-north-1:209479294933:log-group:/aws/codebuild/mip-solver:*"
      ]
    },
    {
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:GetBucketAcl",
        "s3:GetBucketLocation"
      ],
      "Effect": "Allow",
      "Resource": [
        "arn:aws:s3:::codepipeline-eu-north-1-*"
      ]
    },
    {
      "Action": [
        "codebuild:CreateReportGroup",
        "codebuild:CreateReport",
        "codebuild:UpdateReport",
        "codebuild:BatchPutTestCases",
        "codebuild:BatchPutCodeCoverages"
      ],
      "Effect": "Allow",
      "Resource": [
        "arn:aws:codebuild:eu-north-1:209479294933:report-group/mip-solver-*"
      ]
    }
  ],
  "Version": "2012-10-17"
}
POLICY
}

resource "aws_iam_policy" "tfer--CodeBuildCodeConnectionsSourceCredentialsPolicy-mip-solver-eu-north-1" {
  description = "Policy used in trust relationship with CodeBuild"
  name        = "CodeBuildCodeConnectionsSourceCredentialsPolicy-mip-solver-eu-north-1"
  path        = "/service-role/"

  policy = <<POLICY
{
  "Statement": [
    {
      "Action": [
        "codestar-connections:GetConnectionToken",
        "codestar-connections:GetConnection",
        "codeconnections:GetConnectionToken",
        "codeconnections:GetConnection",
        "codeconnections:UseConnection"
      ],
      "Effect": "Allow",
      "Resource": [
        "arn:aws:codestar-connections:eu-north-1:209479294933:connection/836086b3-d1cb-4150-bd23-248edac0db11",
        "arn:aws:codeconnections:eu-north-1:209479294933:connection/836086b3-d1cb-4150-bd23-248edac0db11"
      ]
    }
  ],
  "Version": "2012-10-17"
}
POLICY
}

resource "aws_iam_policy" "tfer--mipServicePolicy" {
  name = "mipServicePolicy"
  path = "/"

  policy = <<POLICY
{
  "Statement": [
    {
      "Action": [
        "sts:AssumeRole",
        "sts:GetCallerIdentity"
      ],
      "Effect": "Allow",
      "Resource": "arn:aws:iam::209479294933:role/mipRole"
    },
    {
      "Action": [
        "s3:*"
      ],
      "Effect": "Allow",
      "Resource": "arn:aws:s3:::lucapolimi-experiments/*"
    },
    {
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Effect": "Allow",
      "Resource": "arn:aws:logs:*:*:*"
    }
  ],
  "Version": "2012-10-17"
}
POLICY
}
