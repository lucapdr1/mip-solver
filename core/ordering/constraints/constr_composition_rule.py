from core.ordering.ordering_rule_interface import OrderingRule
from gurobipy import GRB
from collections import defaultdict
import math

class ConstraintCompositionRule(OrderingRule):
    """
    Assigns a score to each constraint based on whether it involves only integral/binary
    variables, only continuous variables, or a mix.
    
    Scoring logic:
      - Score 3: Constraint involves only integral (or binary/semi-integer) variables.
      - Score 2: Constraint involves only continuous (or semi-continuous) variables.
      - Score 1: Constraint involves a mix of both.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # This rule is for constraints only.
        return [0] * len(vars)

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Iterates over each constraint (row) in A, examines the variables that appear,
        and assigns a category score:
          - 3 if only integral variables appear,
          - 2 if only continuous variables appear,
          - 1 if a mix appears.
        """
        scores = []
        for row_idx, constr in enumerate(constraints):
            row_data = A[row_idx, :]
            if hasattr(row_data, "toarray"):
                row_data = row_data.toarray().flatten()
            nz_indices = row_data.nonzero()[0]

            has_integral = False
            has_continuous = False

            for col_idx in nz_indices:
                var = vars[col_idx]
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
        This rule does not affect variable ordering, so we return a fixed tuple.
        """
        return (0,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Returns the score for a single constraint as a tuple by leveraging score_constraints.
        """
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars, obj_coeffs, bounds, A, [constraints[idx]], rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, constraints, rhs):
        """
        Partitions the block defined by var_indices and constr_indices based on the
        constraint composition score computed on the corresponding submatrix.
        
        Steps:
        1. Construct sub-lists for variables, bounds, constraints, and rhs for the block.
        2. Extract the submatrix of A corresponding to these indices.
        3. Call score_constraints on the sub-lists and submatrix.
        4. Log the computed scores.
        5. Group the original constraint indices by these scores.
        6. Group all variable indices together (since this rule does not affect variables).
        7. Sort the constraint groups by their score in descending order.
        8. Form the partition map as the Cartesian product of these groups.
        
        Returns a dictionary mapping block labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Construct sub-lists for the current block.
        vars_sub = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]  # Not used, but for interface consistency.
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub = [rhs[i] for i in constr_indices] if rhs is not None else None

        # Extract the submatrix corresponding to the block.
        submatrix = A[constr_indices, :][:, var_indices]
        
        # Compute constraint scores on the submatrix.
        sub_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub)
        
        # Group the original constraint indices by their computed scores.
        constr_groups = defaultdict(list)
        for idx, score in zip(constr_indices, sub_scores):
            constr_groups[score].append(idx)
            
        # All variables are grouped together, as this rule does not differentiate them.
        var_groups = {0: list(var_indices)}
        
        # Sort the constraint groups by their score in descending order.
        sorted_scores = sorted(constr_groups.keys(), reverse=True)
        
        # Form the partition map as the Cartesian product of variable groups and the sorted constraint groups.
        partition_map = {}
        label = 0
        for score in sorted_scores:
            partition_map[label] = (list(var_indices), constr_groups[score])
            label += 1
                    
        return partition_map

