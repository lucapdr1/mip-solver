import json
import sys
from core.ordering.constraints.constr_composition_rule import ConstraintCompositionRule
from core.ordering.constraints.constraint_range_rule import ConstraintRangeRule
from core.ordering.constraints.rhs_value_rule import RHSValueRule
from core.ordering.constraints.row_coefficient_rule import RowCoefficientRule
from core.ordering.recursive.cardinality_rule import NonZeroCountRule, ObjectiveNonZeroCountRule, RHSNonZeroCountRule
from core.ordering.recursive.scale_Invariant_rules import BothBoundsFiniteCountRule, BothBoundsInfiniteCountRule, ConstraintContinuousCountRule, ConstraintIntegerCountRule, OneBoundFiniteCountRule
from core.ordering.recursive.sign_pattern_rule import SignPatternRule
from core.ordering.recursive.specific_rules import AllBinaryVariablesRule, AllCoefficientsOneRule, SetPackingRHSRule, UnscaledObjectiveOrderingRule
from core.ordering.variables.cols_bound_category_rule import BoundCategoryRule
from core.ordering.variables.cols_coefficient_rule import ColumnsCoefficientRule
from core.ordering.variables.objective_coefficient_rule import ObjectiveCoefficientRule
from core.ordering.variables.variable_occurrence_rule import VariableOccurrenceRule
from core.ordering.variables.variable_type_rule import VariableTypeRule

RULE_MAP = {
    "VariableTypeRule": VariableTypeRule(),
    "BoundCategoryRule": BoundCategoryRule(),
    "ConstraintCompositionRule": ConstraintCompositionRule(),
    "AllCoefficientsOneRule": AllCoefficientsOneRule(),
    "AllBinaryVariablesRule": AllBinaryVariablesRule(),
    "NonZeroCountRule": NonZeroCountRule(),
    "ObjectiveNonZeroCountRule" : ObjectiveNonZeroCountRule(),
    "RHSNonZeroCountRule" : RHSNonZeroCountRule(),
    "SignPatternRule": SignPatternRule(),
    "ConstraintIntegerCountRule": ConstraintIntegerCountRule(),
    "ConstraintContinuousCountRule": ConstraintContinuousCountRule(),
    "BothBoundsFiniteCountRule": BothBoundsFiniteCountRule(),
    "BothBoundsInfiniteCountRule": BothBoundsInfiniteCountRule(),
    "OneBoundFiniteCountRule": OneBoundFiniteCountRule(),
    "ColumnsCoefficientRule": ColumnsCoefficientRule(1),
    "ObjectiveCoefficientRule": ObjectiveCoefficientRule(100),
    "VariableOccurrenceRule": VariableOccurrenceRule(1),
    "RowCoefficientRule": RowCoefficientRule(1),
    "RHSValueRule": RHSValueRule(100),
    "ConstraintRangeRule": ConstraintRangeRule(1),
    "SetPackingRHSRule" : SetPackingRHSRule(),
    "UnscaledObjectiveOrderingRule" : UnscaledObjectiveOrderingRule()

}

def load_rules_from_json(file_path):
    """Load rules from JSON file and return them as a list."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        rules = []
        for rule_name in data.get("matrix_repatable_rules", []):
            rule_instance = RULE_MAP.get(rule_name)
            if rule_instance:
                rules.append(rule_instance)
            else:
                print(f"Warning: Rule '{rule_name}' is not recognized and will be skipped.")
        return rules
    except Exception as e:
        print(f"Error loading rules from JSON: {e}")
        sys.exit(1)