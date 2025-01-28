from core.ordering.ordering_rule_interface import OrderingRule
from gurobipy import GRB

class VariableTypeRule(OrderingRule):
    def score_variables(self, vars, obj_coeffs, bounds):
        # Prioritize variables by type: Binary, Integer, Continuous
        type_priority = {GRB.BINARY: 3, GRB.INTEGER: 2, GRB.CONTINUOUS: 1}
        return [type_priority.get(var.VType, 0) * 1e3 for var in vars]

    def score_constraints(self, constraints, A, rhs):
        # This rule does not reorder constraints
        return [0] * len(constraints)
