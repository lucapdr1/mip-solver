output "aws_batch_compute_environment_tfer--mip-solver-env_id" {
  value = "${aws_batch_compute_environment.tfer--mip-solver-env.id}"
}

output "aws_batch_compute_environment_tfer--mip-solver-on-demand-env_id" {
  value = "${aws_batch_compute_environment.tfer--mip-solver-on-demand-env.id}"
}

output "aws_batch_job_definition_tfer--mip-solver-job-definition-003A-0xc002172bdc_id" {
  value = "${aws_batch_job_definition.tfer--mip-solver-job-definition-003A-0xc002172bdc.id}"
}

output "aws_batch_job_queue_tfer--mip-solver-queue_id" {
  value = "${aws_batch_job_queue.tfer--mip-solver-queue.id}"
}
