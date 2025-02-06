# main.py

from core.optimization_experiment import OptimizationExperiment
from core.ordering.rule_combination import RuleComposition
from core.ordering.rule_combination_hierarchical import HierarchicalRuleComposition
from core.ordering.variable_type_rule import VariableTypeRule
from core.ordering.cols_bound_category_rule import BoundCategoryRule
from core.ordering.objective_coefficient_rule import ObjectiveCoefficientRule
from core.ordering.cols_coefficient_rule import ColumnsCoefficientRule
from core.ordering.variable_occurrence_rule import VariableOccurrenceRule
from core.ordering.constraint_sense_rule import ConstraintSenseRule
from core.ordering.constr_composition_rule import ConstraintCompositionRule
from core.ordering.rhs_value_rule import RHSValueRule
from core.ordering.row_coefficient_rule import RowCoefficientRule
from core.ordering.constraint_range_rule import ConstraintRangeRule
from utils.config import INPUT_PROBLEM, NUMBER_OF_PERMUATATIONS

<<<<<<< HEAD
from utils.gurobi_utils import init_gurobi_env

rules = [
    #cols
    VariableTypeRule(1e5),
=======
var_block_rules = [
    VariableTypeRule(1),  # e.g., BINARY->3, INTEGER->2, CONTINUOUS->1
    BoundCategoryRule(1)
    # You could add more block rules here
]

var_intra_rules = [
>>>>>>> 560e2f90cee529dda4d73666c4ac1c88efb38abc
    ColumnsCoefficientRule(1),
    ObjectiveCoefficientRule(100),
    VariableOccurrenceRule(1)
]

constr_block_rules = [
    ConstraintSenseRule(1),
    ConstraintCompositionRule(1)
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
    constr_intra_rules=constr_intra_rules
)


if __name__ == "__main__":    
    try:
        gp_env = init_gurobi_env()

        experiment = OptimizationExperiment(gp_env, INPUT_PROBLEM, ordering_rule)
        results = experiment.run_experiment(NUMBER_OF_PERMUATATIONS)
    except Exception as e:
        print(f"Experiment failed: {e}")
