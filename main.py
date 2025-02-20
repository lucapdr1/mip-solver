# main.py

from core.optimization_experiment import OptimizationExperiment
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
from core.ordering.recursive.cardinality_rule import NonZeroCountRule
from core.ordering.recursive.sign_pattern_rule import SignPatternRule
from core.ordering.recursive.scale_Invariant_rules import ConstraintIntegerCountRule, ConstraintContinuousCountRule, BothBoundsFiniteCountRule, BothBoundsInfiniteCountRule, OneBoundFiniteCountRule
from core.ordering.recursive.normalized_occurrence_rule import NormalizedOccurrenceCountRule
from core.ordering.recursive.specific_rules import AllBinaryVariablesRule, AllCoefficientsOneRule
from utils.gurobi_utils import init_gurobi_env, get_Input_problem
from utils.config import NUMBER_OF_PERMUTATIONS, RECURSIVE_RULES


def create_hierarchical_ordering():
    """Old Hierarchical Approach"""
    var_block_rules = [
        VariableTypeRule(1),  
        BoundCategoryRule(1)
    ]
    
    var_intra_rules = [
        ColumnsCoefficientRule(1),
        ObjectiveCoefficientRule(100),
        VariableOccurrenceRule(1)
    ]

    constr_block_rules = [
        ConstraintSenseRule(1),
        ConstraintCompositionRule(1)
    ]

    constr_intra_rules = [
        RowCoefficientRule(1),
        RHSValueRule(100),
        ConstraintRangeRule(1)
    ]

    return HierarchicalRuleComposition(
        var_block_rules=var_block_rules,
        var_intra_rules=var_intra_rules,
        constr_block_rules=constr_block_rules,
        constr_intra_rules=constr_intra_rules
    )


def create_recursive_hierarchical_ordering():
    """New Recursive Hierarchical Approach"""
    matrix_block_rules = [
        VariableTypeRule(),
        BoundCategoryRule(),
        ConstraintCompositionRule(),
    ]

    matrix_repatable_rules = [
        #Rules that likely are producing blocks only on very few instances
        AllCoefficientsOneRule(),
        AllBinaryVariablesRule(),
        #All the other rules
        NonZeroCountRule(),
        SignPatternRule(),
        ConstraintIntegerCountRule(),
        ConstraintContinuousCountRule(),
        BothBoundsFiniteCountRule(),
        BothBoundsInfiniteCountRule(),
        OneBoundFiniteCountRule()
    ]

    matrix_intra_rules = [
        ColumnsCoefficientRule(1),
        ObjectiveCoefficientRule(100),
        VariableOccurrenceRule(1),
        RowCoefficientRule(1),
        RHSValueRule(100),
        ConstraintRangeRule(1)
    ]

    return RecursiveHierarchicalRuleComposition(
        matrix_block_rules_parent=matrix_block_rules,
        matrix_block_rules_child=matrix_repatable_rules,
        #matrix_intra_rules=matrix_intra_rules,
        max_depth=50
    )


if __name__ == "__main__":
    try:
        gp_env = init_gurobi_env()
        input_problem = get_Input_problem()

        ordering_rule = create_recursive_hierarchical_ordering() if RECURSIVE_RULES else create_hierarchical_ordering()

        experiment = OptimizationExperiment(gp_env, input_problem, ordering_rule)
        results = experiment.run_experiment(NUMBER_OF_PERMUTATIONS)
    except Exception as e:
        print(f"Experiment failed: {e}")
