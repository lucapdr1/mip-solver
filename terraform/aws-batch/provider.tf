provider "aws" {}

terraform {
	required_providers {
		aws = {
	    version = "~> 5.94.1"
		}
  }
}
