# main.py

from core.optimization_experiment import OptimizationExperiment
from core.ordering.rule_combination import RuleComposition
from core.ordering.rule_combination_hierarchical import HierarchicalRuleComposition
from core.ordering.rule_combination_recursive import RecursiveHierarchicalRuleComposition
from core.ordering.variables.variable_type_rule import VariableTypeRule
from core.ordering.variables.cols_bound_category_rule import BoundCategoryRule
from core.ordering.variables.objective_coefficient_rule import ObjectiveCoefficientRule
from core.ordering.variables.cols_coefficient_rule import ColumnsCoefficientRule
from core.ordering.variables.variable_occurrence_rule import VariableOccurrenceRule
from core.ordering.constraints.constraint_sense_rule import ConstraintSenseRule
from core.ordering.constraints.constr_composition_rule import ConstraintCompositionRule
from core.ordering.constraints.rhs_value_rule import RHSValueRule
from core.ordering.constraints.row_coefficient_rule import RowCoefficientRule
from core.ordering.constraints.constraint_range_rule import ConstraintRangeRule
from core.ordering.recursive.cardinality_rule import CardinalityRule
from core.ordering.recursive.sign_pattern_rule import SignPatternRule
from core.ordering.recursive.scale_Invariant_rules import MaxMinRatioRule, ConstraintIntegerCountRule, ConstraintContinuousCountRule, ObjectiveToColumnSumRatioRule, RHSToRowSumRatioRule
from core.ordering.recursive.normalized_occurrence_rule import NormalizedOccurrenceCountRule
from utils.gurobi_utils import init_gurobi_env, get_Input_problem
from utils.config import NUMBER_OF_PERMUATATIONS


var_block_rules = [
    #VariableTypeRule(1),  # e.g., BINARY->3, INTEGER->2, CONTINUOUS->1
    #BoundCategoryRule(1),
    #SignPatternRule(),
    CardinalityRule(),
    #MaxMinRatioRule(),
    #ObjectiveToColumnSumRatioRule()
    # You could add more block rules here
]

var_intra_rules = [
    #ColumnsCoefficientRule(1),
    #ObjectiveCoefficientRule(100),
    #VariableOccurrenceRule(1)
]

constr_block_rules = [
    #ConstraintSenseRule(1),
    #ConstraintCompositionRule(1),
    #SignPatternRule(),
    CardinalityRule(),
    #MaxMinRatioRule(),
    #ConstraintIntegerCountRule(),
    #ConstraintContinuousCountRule(),
    #RHSToRowSumRatioRule()
    # Possibly more block rules
]

constr_intra_rules = [
    #RowCoefficientRule(1),
    #RHSValueRule(100),
    #ConstraintRangeRule(1)
]

"""
# Old Hierarchical approach
ordering_rule = HierarchicalRuleComposition(
    var_block_rules=var_block_rules,
    var_intra_rules=var_intra_rules,
    constr_block_rules=constr_block_rules,
    constr_intra_rules=constr_intra_rules
)
"""
matrix_block_rules =  [
    VariableTypeRule(),
    BoundCategoryRule(),
]

matrix_repatable_rules = [
    ConstraintCompositionRule(),
    CardinalityRule(),
    SignPatternRule(),
    #NormalizedOccurrenceCountRule()
]

ordering_rule = RecursiveHierarchicalRuleComposition(
    matrix_block_rules_parent=matrix_block_rules,
    matrix_block_rules_child=matrix_repatable_rules,
    max_depth=50  # limit recursion depth
)

if __name__ == "__main__":    
    try:
        gp_env = init_gurobi_env()
        input_problem = get_Input_problem()
        experiment = OptimizationExperiment(gp_env, input_problem, ordering_rule)
        results = experiment.run_experiment(NUMBER_OF_PERMUATATIONS)
    except Exception as e:
        print(f"Experiment failed: {e}")
