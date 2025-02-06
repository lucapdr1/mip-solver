from core.ordering.ordering_rule_interface import OrderingRule
from gurobipy import GRB

class ConstraintCompositionRule(OrderingRule):
    """
    Assigns a score to each constraint based on whether it
    involves only integral/binary variables, only continuous,
    or a mix.

    Example:
      - 3 => only integral or binary
      - 2 => only continuous
      - 1 => mix of both
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # No effect on variables
        return [0] * len(vars)

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # We'll iterate over each constraint row,
        # check the variables that appear, and categorize
        scores = []
        for row_idx, constr in enumerate(constraints):
            row_data = A[row_idx, :]
            nz_indices = row_data.nonzero()[0]

            has_integral = False
            has_continuous = False

            for col_idx in nz_indices:
                var = vars[col_idx]  # Directly use the passed vars list

                if var.VType in [GRB.BINARY, GRB.INTEGER, GRB.SEMIINT]:
                    has_integral = True
                elif var.VType in [GRB.CONTINUOUS, GRB.SEMICONT]:
                    has_continuous = True

                if has_integral and has_continuous:
                    break

            if has_integral and not has_continuous:
                cat = 3
            elif has_continuous and not has_integral:
                cat = 2
            else:
                cat = 1

            scores.append(cat * self.scaling)

        return scores