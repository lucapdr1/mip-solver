from core.ordering.ordering_rule_interface import OrderingRule

class ConstraintSenseRule(OrderingRule):
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # This rule does not reorder variables
        return [0] * len(vars)

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # Prioritize constraints by type: '<', '=', '>'
        #TODO: check >= etc
        sense_priority = {'<': 1, '=': 2, '>': 3}
        return [sense_priority.get(c.Sense, 0) * self.scaling for c in constraints]
