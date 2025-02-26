from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelInfo:
    file_name: str
    instance: Optional[str] = None
    variables: Optional[int] = None
    constraints: Optional[int] = None
    objective_sense: Optional[str] = None

@dataclass
class IterationMetrics:
    models_equivalent: bool
    variable_counts_match: bool
    constraint_counts_match: bool
    original_objective_value: float
    permuted_objective_value: float
    canonical_original_objective_value: float
    canonical_permuted_objective_value: float
    original_solve_time: float
    permuted_solve_time: float
    canonical_original_solve_time: float
    canonical_permuted_solve_time: float
    original_work_units: float
    permuted_work_units: float
    canonical_original_work_units: float
    canonical_permuted_work_units: float
    permutation_distance_before: float
    permutation_distance_after: float

@dataclass
class AggregatedMetrics:
    std_all_permutation_solve_time: Optional[float] = None
    std_all_canonical_solve_time: Optional[float] = None
    std_all_permutation_work_units: Optional[float] = None
    std_all_canonical_work_units: Optional[float] = None

    std_perm_distance_before: Optional[float] = None
    std_perm_distance_after: Optional[float] = None

    models_equivalent_pct: Optional[float] = None
    variable_counts_match_pct: Optional[float] = None
    constraint_counts_match_pct: Optional[float] = None

@dataclass
class GranularityStats:
    total_blocks: Optional[int] = None
    avg_block_size: Optional[float] = None
    min_block_size: Optional[int] = None
    max_block_size: Optional[int] = None
    sum_subblocks_sizes: Optional[int] = None
    original_matrix_size: Optional[int] = None
@dataclass
class AggregatedGranularityMetrics:
    avg_block_percentage : Optional[float] = None

@dataclass
class LogMetrics:
    model_info: ModelInfo
    iterations: List[IterationMetrics] = field(default_factory=list)
    aggregated_metrics: AggregatedMetrics = field(default_factory=AggregatedMetrics)
    granularity_stats: Optional[GranularityStats] = None 
    granularity_metrics : Optional[AggregatedGranularityMetrics] = None