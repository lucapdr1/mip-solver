from core.ordering.ordering_rule_interface import OrderingRule

class ConstraintSenseRule(OrderingRule):
    def score_variables(self, vars, obj_coeffs, bounds):
        # This rule does not reorder variables
        return [0] * len(vars)

    def score_constraints(self, constraints, A, rhs):
        # Prioritize constraints by type: '<', '=', '>'
        #TODO: check >= etc
        sense_priority = {'<': 1, '=': 2, '>': 3}
        return [sense_priority.get(c.Sense, 0) * 1e3 for c in constraints]
