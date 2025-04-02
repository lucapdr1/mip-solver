# main.py

import sys
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
from core.ordering.recursive.cardinality_rule import NonZeroCountRule, ObjectiveNonZeroCountRule, RHSNonZeroCountRule
from core.ordering.recursive.sign_pattern_rule import SignPatternRule
from core.ordering.recursive.scale_Invariant_rules import ConstraintIntegerCountRule, ConstraintContinuousCountRule, BothBoundsFiniteCountRule, BothBoundsInfiniteCountRule, OneBoundFiniteCountRule
from core.ordering.recursive.normalized_occurrence_rule import NormalizedOccurrenceCountRule
from core.ordering.recursive.specific_rules import AllBinaryVariablesRule, AllCoefficientsOneRule, SetPackingRHSRule, UnscaledObjectiveOrderingRule
from core.ordering.recursive.ladder_intra_rule import LadderIntraRule
from core.ordering.recursive.adjacency_aware import ReverseCuthillMcKeeRule, AdjacencyClusteringRule
from core.ordering.constraints.decomposition_rule import DecompositionRule
from utils.gurobi_utils import init_gurobi_env, get_Input_problem
from utils.rulemap import load_rules_from_json
from utils.dec_parser import DecFileParser
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
        AdjacencyClusteringRule(),
        ReverseCuthillMcKeeRule(),
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


def create_recursive_hierarchical_ordering(input_problem, json_file=None):
    dec_parser = DecFileParser(input_problem)

    """New Recursive Hierarchical Approach"""
    matrix_block_rules = [
        DecompositionRule(dec_parser=dec_parser)
        #VariableTypeRule(),
        #BoundCategoryRule(),
        #ConstraintCompositionRule(),
    ]

    if json_file:
        print(f"Loading matrix_repeatable_rules from {json_file}...")
        matrix_repatable_rules = load_rules_from_json(json_file)
    else:
        matrix_repatable_rules = [
            #Rules that likely are producing blocks only on very few instances
            #AllBinaryVariablesRule(),
            #AllCoefficientsOneRule(),
            ###All the other rules
            #NonZeroCountRule(),
            ##ObjectiveNonZeroCountRule(),
            ##RHSNonZeroCountRule(),
            #SignPatternRule(),
            #ConstraintIntegerCountRule(),
            #ConstraintContinuousCountRule(),
            #BothBoundsFiniteCountRule(),
            #BothBoundsInfiniteCountRule(),
            #OneBoundFiniteCountRule(),
            ##Specific for setpacking and setcovering
            #SetPackingRHSRule(),
            #UnscaledObjectiveOrderingRule(),
            
        ]

    matrix_intra_rules = [
        #LadderIntraRule(0.3),
        #ColumnsCoefficientRule(1),
        #ObjectiveCoefficientRule(100),
        #VariableOccurrenceRule(1),
        #RowCoefficientRule(1),
        #RHSValueRule(100),
        #ConstraintRangeRule(1)
    ]

    return RecursiveHierarchicalRuleComposition(
        matrix_block_rules_parent=matrix_block_rules,
        matrix_block_rules_child=matrix_repatable_rules,
        matrix_intra_rules=matrix_intra_rules,
        max_depth=50
    )


if __name__ == "__main__":
    try:
        gp_env = init_gurobi_env()
        input_problem = get_Input_problem()

        # Check if a JSON file was passed as an argument
        json_file = sys.argv[1] if len(sys.argv) > 1 else None

        ordering_rule = create_recursive_hierarchical_ordering(input_problem, json_file) if RECURSIVE_RULES else create_hierarchical_ordering()

        experiment = OptimizationExperiment(gp_env, input_problem, ordering_rule)
        results = experiment.run_experiment(NUMBER_OF_PERMUTATIONS)
    except Exception as e:
        print(f"Experiment failed: {e}")
