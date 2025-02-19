import re
from typing import List
from log_parser.model import ModelInfo, IterationMetrics, AggregatedMetrics, LogMetrics

def parse_model_info(content: str, file_name: str) -> ModelInfo:
    """Extract general model information from the log content."""
    info = ModelInfo(file_name=file_name)
    instance_match = re.search(r'Successfully loaded problem from S3: .*/(.*?)\.mps', content)
    if instance_match:
        info.instance = instance_match.group(1)
    
    prob_details = re.search(
        r'Problem Details:.*?- Variables: (\d+).*?- Constraints: (\d+).*?- Objective Sense: (\w+)',
        content, re.DOTALL
    )
    if prob_details:
        info.variables = int(prob_details.group(1))
        info.constraints = int(prob_details.group(2))
        info.objective_sense = prob_details.group(3)
    
    return info

def parse_iterations(content: str) -> List[IterationMetrics]:
    """Extract per-iteration metrics from the log content."""
    iterations = []
    pattern = re.compile(
        r'Running iteration (\d+)/\d+.*?'
        r'- Models equivalent: (True|False).*?'
        r'- Variable counts match: (True|False).*?'
        r'- Constraint counts match: (True|False).*?'
        r'- Original Objective Value: ([-+]?[\d.]+).*?'
        r'- Permuted Objective Value: ([-+]?[\d.]+).*?'
        r'- Canonical from Original Objective Value: ([-+]?[\d.]+).*?'
        r'- Canonical from Permuted Objective Value: ([-+]?[\d.]+).*?'
        r'- Original Solve Time: ([\d.]+) seconds.*?'
        r'- Permuted Solve Time: ([\d.]+) seconds.*?'
        r'- Canonical from Original Solve Time: ([\d.]+) seconds.*?'
        r'- Canonical from Permuted Solve Time: ([\d.]+) seconds.*?'
        r'- Original Work Units: ([\d.]+).*?'
        r'- Permuted Work Units: ([\d.]+).*?'
        r'- Canonical from Original Work Units: ([\d.]+).*?'
        r'- Canonical from Permuted Work Units: ([\d.]+).*?'
        r'- Permutation Distance Before Canonicalization: ([\d.]+).*?'
        r'- Permutation Distance After Canonicalization: ([\d.]+)',
        re.DOTALL
    )
    
    for match in pattern.finditer(content):
        iteration = IterationMetrics(
            models_equivalent = match.group(2) == 'True',
            variable_counts_match = match.group(3) == 'True',
            constraint_counts_match = match.group(4) == 'True',
            original_objective_value = float(match.group(5)),
            permuted_objective_value = float(match.group(6)),
            canonical_original_objective_value = float(match.group(7)),
            canonical_permuted_objective_value = float(match.group(8)),
            original_solve_time = float(match.group(9)),
            permuted_solve_time = float(match.group(10)),
            canonical_original_solve_time = float(match.group(11)),
            canonical_permuted_solve_time = float(match.group(12)),
            original_work_units = float(match.group(13)),
            permuted_work_units = float(match.group(14)),
            canonical_original_work_units = float(match.group(15)),
            canonical_permuted_work_units = float(match.group(16)),
            permutation_distance_before = float(match.group(17)),
            permutation_distance_after = float(match.group(18))
        )
        iterations.append(iteration)
    return iterations