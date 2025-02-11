from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from collections import defaultdict

class SignPatternRule(OrderingRule):
    """
    Classifies the sign pattern of the nonzero coefficients:
      - For columns (variables): whether all coefficients are >= 0, all <= 0, or mixed.
      - For rows (constraints): the same check for each row.
    
    Returns integer 'pattern codes':
      - 2 => all nonnegative
      - 1 => all nonpositive
      - 0 => mixed (or all zero)
    """

    def __init__(self):
        pass  # No scaling parameter needed

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Determine sign pattern for each variable's column in A.
        """
        num_constraints, num_vars = A.shape
        scores = np.zeros(num_vars, dtype=int)

        for j in range(num_vars):
            column = A[:, j]
            if hasattr(column, "toarray"):
                column = column.toarray().flatten()  # Convert sparse to dense

            has_pos = np.any(column > 1e-15)
            has_neg = np.any(column < -1e-15)

            if has_pos and not has_neg:
                scores[j] = 2  # All positive
            elif has_neg and not has_pos:
                scores[j] = 1  # All negative
            else:
                scores[j] = 0  # Mixed or all zero

        return scores.tolist()

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Determine sign pattern for each constraint's row in A.
        """
        num_constraints, num_vars = A.shape
        scores = np.zeros(num_constraints, dtype=int)

        for i in range(num_constraints):
            row = A[i, :]
            if hasattr(row, "toarray"):
                row = row.toarray().flatten()
            
            has_pos = np.any(row > 1e-15)
            has_neg = np.any(row < -1e-15)

            if has_pos and not has_neg:
                scores[i] = 2  # All positive
            elif has_neg and not has_pos:
                scores[i] = 1  # All negative
            else:
                scores[i] = 0  # Mixed or all zero

        return scores.tolist()

    # --- Methods for Rectangular Block Reordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Computes the sign pattern for a single variable (column) and returns it as a tuple.
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Computes the sign pattern for a single constraint (row) and returns it as a tuple.
        """
        rhs_single = np.array([rhs[idx]]) if rhs is not None else None
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
          3. Compute variable scores on the submatrix by calling score_variables.
          4. Compute constraint scores on the submatrix by calling score_constraints.
          5. Group the original indices by these computed scores.
          6. Form the partition as the Cartesian product of the variable groups and constraint groups.
        
        Returns a dictionary mapping block labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Construct sub-lists for the current block.
        vars_sub = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub = [rhs[i] for i in constr_indices] if rhs is not None else None

        # Extract the submatrix corresponding to the current block.
        submatrix = A[constr_indices, :][:, var_indices]

        # Compute scores on the submatrix using existing methods.
        sub_var_scores = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)

        # Group original variable indices by their computed scores.
        var_groups = defaultdict(list)
        for idx, score in zip(var_indices, sub_var_scores):
            var_groups[score].append(idx)

        # Group constraints similarly.
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
