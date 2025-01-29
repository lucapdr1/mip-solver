# main.py

from core.optimization_experiment import OptimizationExperiment
from core.ordering.rule_combination import RuleComposition
from core.ordering.variable_type_rule import VariableTypeRule
from core.ordering.objective_coefficient_rule import ObjectiveCoefficientRule
from core.ordering.cols_coefficient_rule import ColumnsCoefficientRule
from core.ordering.variable_occurrence_rule import VariableOccurrenceRule
from core.ordering.constraint_sense_rule import ConstraintSenseRule
from core.ordering.rhs_value_rule import RHSValueRule
from core.ordering.row_coefficient_rule import RowCoefficientRule
from core.ordering.constraint_range_rule import ConstraintRangeRule
from utils.config import INPUT_PROBLEM

rules = [
    #cols
    VariableTypeRule(1e3),
    ObjectiveCoefficientRule(100),
    ColumnsCoefficientRule(10),
    VariableOccurrenceRule(),
    #rows
    ConstraintSenseRule(1e3),
    RHSValueRule(100),
    RowCoefficientRule(10),
    ConstraintRangeRule()
    

]
ordering_rule = RuleComposition(rules)

if __name__ == "__main__":    
    try:
        experiment = OptimizationExperiment(INPUT_PROBLEM, ordering_rule)
        results = experiment.run_experiment(4)
    except Exception as e:
        print(f"Experiment failed: {e}")
