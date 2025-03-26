import re
from typing import List, Optional
from log_parser.model import ModelInfo, IterationMetrics, GranularityStats, PairwiseDistances

def parse_model_info(content: str, file_name: str) -> ModelInfo:
    """Extract general model information from the log content."""
    info = ModelInfo(file_name=file_name)
    instance_match = re.search(
    r"Successfully loaded (?:local )?problem from (?:S3:\s*)?(?:\./)?(?:.*[\\/])?([^\\/]+?)\.mps",
    content)
    
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


def parse_float_or_none(val: str) -> Optional[float]:
    """
    Converts the provided string to a float.
    Returns None if the string is "None" or cannot be converted.
    """
    if val.strip() == "None":
        return None
    try:
        return float(val)
    except ValueError:
        return None

def parse_iterations(content: str) -> List[IterationMetrics]:
    """
    Extracts per-iteration metrics from the log content.
    This version handles cases where objective values (and possibly others)
    may be 'None' rather than a numeric value.
    """
    iterations = []
    pattern = re.compile(
        r'Iteration\s+\d+\s+Results:.*?'
        r'- Models equivalent:\s*(True|False).*?'
        r'- Variable counts match:\s*(True|False).*?'
        r'- Constraint counts match:\s*(True|False).*?'
        r'- Original Objective Value:\s*(None|[-+]?[\d.]+).*?'
        r'- Permuted Objective Value:\s*(None|[-+]?[\d.]+).*?'
        r'- Canonical from Original Objective Value:\s*(None|[-+]?[\d.]+).*?'
        r'- Canonical from Permuted Objective Value:\s*(None|[-+]?[\d.]+).*?'
        r'- Original Solve Time:\s*([\d.]+)\s*seconds.*?'
        r'- Permuted Solve Time:\s*([\d.]+)\s*seconds.*?'
        r'- Canonical from Original Solve Time:\s*([\d.]+)\s*seconds.*?'
        r'- Canonical from Permuted Solve Time:\s*([\d.]+)\s*seconds.*?'
        r'- Original Work Units:\s*([\d.]+).*?'
        r'- Permuted Work Units:\s*([\d.]+).*?'
        r'- Canonical from Original Work Units:\s*([\d.]+).*?'
        r'- Canonical from Permuted Work Units:\s*([\d.]+).*?'
        r'- Permutation Distance Before Canonicalization:\s*([\d.]+).*?'
        r'- Permutation Distance After Canonicalization:\s*([\d.]+)',
        re.DOTALL
    )
    
    for match in pattern.finditer(content):
        iteration = IterationMetrics(
            models_equivalent = match.group(1) == 'True',
            variable_counts_match = match.group(2) == 'True',
            constraint_counts_match = match.group(3) == 'True',
            original_objective_value = parse_float_or_none(match.group(4)),
            permuted_objective_value = parse_float_or_none(match.group(5)),
            canonical_original_objective_value = parse_float_or_none(match.group(6)),
            canonical_permuted_objective_value = parse_float_or_none(match.group(7)),
            original_solve_time = float(match.group(8)),
            permuted_solve_time = float(match.group(9)),
            canonical_original_solve_time = float(match.group(10)),
            canonical_permuted_solve_time = float(match.group(11)),
            original_work_units = float(match.group(12)),
            permuted_work_units = float(match.group(13)),
            canonical_original_work_units = float(match.group(14)),
            canonical_permuted_work_units = float(match.group(15)),
            permutation_distance_before = float(match.group(16)),
            permutation_distance_after = float(match.group(17))
        )
        iterations.append(iteration)
    return iterations

def parse_pairwise_distances(content: str) -> Optional[PairwiseDistances]:
    """
    Parses the pairwise distances statistics block from the log content.
    This block contains the standard deviation for all-pairs distances
    before and after canonicalization.
    """
    pattern = re.compile(
        r'All-Pairs Permutation Distance Variability \(Standard Deviation\):.*?'
        r'- Std\(All-Pairs Distance Before Canonicalization\):\s*([\d.]+).*?'
        r'- Std\(All-Pairs Distance After Canonicalization\):\s*([\d.]+)',
        re.DOTALL
    )
    match = pattern.search(content)
    if match:
        return PairwiseDistances(
            std_pairwise_distance_before=float(match.group(1)),
            std_pairwise_distance_after=float(match.group(2))
        )
    return None

def parse_granularity_stats(content: str) -> Optional[GranularityStats]:
    """
    Parses the granularity statistics block from the log content.
    This block is optional.
    """
    pattern = re.compile(
        r'Granularity Statistics:.*?'
        r'- Total Blocks:\s*(\d+).*?'
        r'- Avg\(Block Size\):\s*([\d.]+).*?'
        r'- Min\(Block Size\):\s*(\d+).*?'
        r'- Max\(Block Size\):\s*(\d+).*?'
        r'- Sum of SubBlocks sizes:\s*(\d+).*?'
        r'- Original matrix size:\s*(\d+)',
        re.DOTALL
    )
    match = pattern.search(content)
    if match:
        return GranularityStats(
            total_blocks=int(match.group(1)),
            avg_block_size=float(match.group(2)),
            min_block_size=int(match.group(3)),
            max_block_size=int(match.group(4)),
            sum_subblocks_sizes=int(match.group(5)),
            original_matrix_size=int(match.group(6))
        )
    return None