from core.ordering.ordering_rule_interface import OrderingRule
from gurobipy import GRB
import math

class VariableTypeRule(OrderingRule):
    """
    Extended variable-type rule that:
      1) Distinguishes semi-continuous or semi-integer variables.
      2) Treats integer variables with [a, a+1] as effectively binary.
      3) Otherwise uses standard binary=3, integer=2, continuous=1.

    Feel free to adjust the numeric scores if you want to reorder
    the "semi-continuous" vs. "binary" vs. "integer" hierarchy.
    """

    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a list of scores, one per variable, reflecting their "type priority."
        Higher score => higher priority in a descending or lexicographic sort.

        The logic is as follows:
          - If var.VType is SEMICONT or SEMIINT => 4
          - Else if var.VType == BINARY => 3
          - Else if var.VType == INTEGER:
                check if (upper_bound - lower_bound) == 1 => 3 (treat as binary)
                else => 2
          - Else if var.VType == CONTINUOUS => 1
          - Else => 0 (fallback)
        """
        scores = []
        for i, var in enumerate(vars):
            lb, ub = bounds[i]  # [lower_bound, upper_bound]
            
            # Check if it's a semi-continuous or semi-integer
            if var.VType in [GRB.SEMICONT, GRB.SEMIINT]:
                score = 4

            # If "true" binary as indicated by Gurobi
            elif var.VType == GRB.BINARY:
                score = 3

            # If general integer, check if effectively binary
            elif var.VType == GRB.INTEGER:
                # We'll interpret "effectively binary" as exactly 1 integer step
                # between lb and ub (e.g., [a,a+1]).
                # You can add rounding if needed for floating bounds.
                if math.isclose(ub - lb, 1.0, abs_tol=1e-9):
                    score = 3  # treat as binary
                else:
                    score = 2

            # If continuous
            elif var.VType == GRB.CONTINUOUS:
                score = 1

            else:
                # Fallback if some other type is encountered
                score = 0

            scores.append(score * self.scaling)

        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # This rule focuses on variable type, so it does not reorder constraints
        return [0] * len(constraints)
