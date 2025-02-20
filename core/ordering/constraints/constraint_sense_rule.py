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
    
    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the score for a single variable (column) as a one-element tuple.
        Since this rule does not assign meaningful scores to variables, it always returns (0,).
        """
        return (0,)
    
    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the sign-based score for a single constraint (row) as a one-element tuple.
        """
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars,
                                       obj_coeffs,
                                       bounds,
                                       A,
                                       [constraints[idx]],
                                       rhs_single)[0]
        return (score,)
