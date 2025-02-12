from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from collections import defaultdict

class SignPatternRule(OrderingRule):
    """
    Modified rule that extracts sign information only from constraints.
    
    For **variables (columns)**:
      - Since the sign pattern is not invariant (due to possible row scaling
        and multiplication by -1), we do not attempt to extract a sign pattern.
        Instead, we return a constant score (0) for each variable.
    
    For **constraints (rows)**:
      - We count the number of positive and negative entries (using a tolerance).
      - Then, we adjust these counts by adding one to the count corresponding
        to the slack variable’s sign (based on the inequality operator in the constraint).
        For example, for a "≤" constraint we add one to the positive count (since
        the slack is nonnegative), and for a "≥" constraint we add one to the negative count.
      - The final score is the maximum of these two adjusted counts.
    """

    def __init__(self):
        self.tol = 1e-15  # tolerance for considering a value nonzero

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Since sign information for columns is not invariant under row scaling,
        we do not try to compute it. Instead, we simply return a constant score,
        for example 0 for each variable.
        """
        num_constraints, num_vars = A.shape
        return [0] * num_vars

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        For each constraint (row), count the positive and negative entries
        and adjust the count by adding one based on the slack variable's sign.
        The final score is the maximum of the two (adjusted) counts.
        """
        num_constraints, num_vars = A.shape
        scores = np.zeros(num_constraints, dtype=int)

        for i in range(num_constraints):
            row = A[i, :]
            if hasattr(row, "toarray"):
                row = row.toarray().flatten()
            else:
                row = np.ravel(row)

            pos_count = np.sum(row > self.tol)
            neg_count = np.sum(row < -self.tol)

            # Determine slack adjustment from the constraint operator.
            slack_sign = None
            constr_str = str(constraints[i])
            if "<=" in constr_str or "<" in constr_str:
                slack_sign = "+"
            elif ">=" in constr_str or ">" in constr_str:
                slack_sign = "-"

            # Adjust counts based on the slack's expected sign.
            if slack_sign == "+":
                pos_count += 1
            elif slack_sign == "-":
                neg_count += 1

            scores[i] = max(pos_count, neg_count)

        return scores.tolist()

    # --- Methods for Rectangular Block Reordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the (constant) score for a single variable (column) as a one-element tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, constraints, rhs)[0]
        return (score,)

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

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions the block (defined by var_indices and constr_indices) using sign pattern-based grouping.
        
        The process is:
          1. Construct sub-lists for variables, bounds, and constraints corresponding to the block.
          2. Extract the submatrix corresponding to these indices.
          3. Compute variable scores (which will all be 0) and constraint scores using the above methods.
          4. Group the original indices by these computed scores.
          5. Form the partition as the Cartesian product of the variable groups and constraint groups.
        
        Returns a dictionary mapping block labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Construct sub-lists for the current block.
        vars_sub   = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub    = [rhs[i] for i in constr_indices] if rhs is not None else None

        # Extract the submatrix corresponding to the current block.
        submatrix = A[constr_indices, :][:, var_indices]

        # Compute scores on the submatrix.
        sub_var_scores   = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)

        # Group original variable indices by their computed scores.
        var_groups = defaultdict(list)
        for idx, score in zip(var_indices, sub_var_scores):
            var_groups[score].append(idx)

        # Group original constraint indices by their computed scores.
        constr_groups = defaultdict(list)
        for idx, score in zip(constr_indices, sub_constr_scores):
            constr_groups[score].append(idx)

        # Form the partition map as the Cartesian product of the groups.
        partition_map = {}
        label = 0
        for score_v, vgroup in var_groups.items():
            for score_c, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1

        return partition_map
