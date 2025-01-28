from core.ordering.ordering_rule_interface import OrderingRule

class VariableTypeRule(OrderingRule):
    def score_variables(self, vars, obj_coeffs, bounds):
        # Score integer variables higher than continuous
        return [1 if var.VType in ['I', 'B'] else 0 for var in vars]

    def score_constraints(self, constraints, A, rhs):
        # This rule does not reorder constraints
        return [0] * len(constraints)
