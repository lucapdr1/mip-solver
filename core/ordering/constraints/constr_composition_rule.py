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
    
    # --- Methods to Support Rectangular Block/Intra Reordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        This rule does not affect variable ordering. To be compatible with the block
        reordering framework, we return a fixed tuple (e.g., (0,)).
        """
        return (0,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns a tuple containing the score for a single constraint. We leverage the
        existing score_constraints method by calling it on a singleton list.
        """
        # Prepare a single-element rhs if provided.
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars, obj_coeffs, bounds, A, [constraints[idx]], rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions the block (var_indices, constr_indices) based on the constraint composition
        score. Since this rule only affects constraints, we partition as follows:
          - Compute a score for each constraint in constr_indices.
          - Group constraints by the unique score.
          - Group variables together in a single group (since they are unaffected by this rule).
        
        Returns a dictionary mapping block labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        from collections import defaultdict

        # Compute a score for each constraint (as a one-element tuple; we extract the number).
        constr_scores = {}
        for i in constr_indices:
            score_tuple = self.score_matrix_for_constraint(i, vars, obj_coeffs, bounds, A, constraints, rhs)
            constr_scores[i] = score_tuple[0]

        # Group constraints by their score.
        constr_groups = defaultdict(list)
        for idx, s in constr_scores.items():
            constr_groups[s].append(idx)

        # For variables, since this rule does not affect them, we simply put them in one group.
        var_groups = {0: list(var_indices)}

        # Form the partition as the Cartesian product of the variable groups and constraint groups.
        partition_map = {}
        label = 0
        for vscore, vgroup in var_groups.items():
            for cscore, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1

        return partition_map