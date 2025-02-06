import math
from core.ordering.ordering_rule_interface import OrderingRule

class BoundCategoryRule(OrderingRule):
    """
    Scores variables based on the "category" of their bounds:
      4 -> Both bounds finite and nonnegative or nonpositive
      3 -> Both bounds finite and straddle zero (l<0<u)
      2 -> Exactly one bound is infinite
      1 -> Both bounds are infinite
      0 -> fallback, if needed

    You can invert or tweak the hierarchy if you prefer a different order.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        scores = []
        for i, var in enumerate(vars):
            lb, ub = bounds[i]  # (lower_bound, upper_bound)

            # Check infinite bounds
            is_lb_inf = math.isinf(lb)
            is_ub_inf = math.isinf(ub)

            if (not is_lb_inf) and (not is_ub_inf):
                # Both bounds finite
                if lb >= 0 or ub <= 0:
                    # Entire range is nonnegative or nonpositive
                    cat = 4
                elif lb < 0 < ub:
                    # straddles zero
                    cat = 3
                else:
                    # fallback if needed (or combine with above logic)
                    cat = 2
            elif (is_lb_inf and not is_ub_inf) or (not is_lb_inf and is_ub_inf):
                # exactly one infinite bound
                cat = 2
            elif (is_lb_inf and is_ub_inf):
                # both infinite
                cat = 1
            else:
                cat = 0

            scores.append(cat * self.scaling)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        return [0] * len(constraints)