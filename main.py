# main.py

from core.optimization_experiment import OptimizationExperiment
from core.ordering.rule_combination import RuleComposition
from core.ordering.rule_combination_hierarchical import HierarchicalRuleComposition
from core.ordering.variable_type_rule import VariableTypeRule
from core.ordering.objective_coefficient_rule import ObjectiveCoefficientRule
from core.ordering.cols_coefficient_rule import ColumnsCoefficientRule
from core.ordering.variable_occurrence_rule import VariableOccurrenceRule
from core.ordering.constraint_sense_rule import ConstraintSenseRule
from core.ordering.rhs_value_rule import RHSValueRule
from core.ordering.row_coefficient_rule import RowCoefficientRule
from core.ordering.constraint_range_rule import ConstraintRangeRule
from utils.config import INPUT_PROBLEM, NUMBER_OF_PERMUATATIONS

var_block_rules = [
    VariableTypeRule(1),  # e.g., BINARY->3, INTEGER->2, CONTINUOUS->1
    # You could add more block rules here
]

var_intra_rules = [
    ColumnsCoefficientRule(1),
    ObjectiveCoefficientRule(100),
    VariableOccurrenceRule(1)
]

constr_block_rules = [
    ConstraintSenseRule(1),
    # Possibly more block rules
]

constr_intra_rules = [
    RowCoefficientRule(1),
    RHSValueRule(100),
    ConstraintRangeRule(1)
]

ordering_rule = HierarchicalRuleComposition(
    var_block_rules=var_block_rules,
    var_intra_rules=var_intra_rules,
    constr_block_rules=constr_block_rules,
    constr_intra_rules=constr_intra_rules,
    block_rank_factor=1e6
)


if __name__ == "__main__":    
    try:
        experiment = OptimizationExperiment(INPUT_PROBLEM, ordering_rule)
        results = experiment.run_experiment(NUMBER_OF_PERMUATATIONS)
    except Exception as e:
        print(f"Experiment failed: {e}")
