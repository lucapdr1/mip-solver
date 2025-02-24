from core.ordering.ordering_rule_interface import OrderingRule
from gurobipy import GRB
from collections import defaultdict
import numpy as np
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

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # This rule is for constraints only.
       return np.zeros(len(vars), dtype=int)

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Iterates over each constraint (row) in A and assigns a score based on the types of variables
        present:
        - 3 if only integral variables appear,
        - 2 if only continuous variables appear,
        - 1 if a mix appears.
        This version leverages NumPy for inner-loop operations and returns a NumPy array.
        """
        # Precompute an array mapping each variable to a code:
        # 3 for integral/binary/semiint, 2 for continuous/semicont.
        vtype_map = {
            GRB.BINARY: 3,
            GRB.INTEGER: 3,
            GRB.SEMIINT: 3,
            GRB.CONTINUOUS: 2,
            GRB.SEMICONT: 2
        }
        # Precompute the codes for all variables.
        var_codes = np.array([vtype_map[v.VType] for v in vars])
        
        nrows = len(constraints)
        scores = np.empty(nrows, dtype=int)

        for i in range(nrows):
            # Get the nonzero indices of the current row.
            row = A_csr.getrow(i)
            nz_indices = row.indices  # Already a NumPy array.
            if nz_indices.size == 0:
                # If the row has no nonzeros, assign a default score.
                scores[i] = 1 * self.scaling
            else:
                row_codes = var_codes[nz_indices]
                # Use vectorized operations:
                if np.all(row_codes == 3):
                    scores[i] = 3 * self.scaling
                elif np.all(row_codes == 2):
                    scores[i] = 2 * self.scaling
                else:
                    scores[i] = 1 * self.scaling
        return scores

    # --- Methods to Support Rectangular Block/Intra Reordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        This rule does not affect variable ordering, so we return a fixed tuple.
        """
        return (0,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns the score for a single constraint as a tuple by leveraging score_constraints.
        """
        # Create a one-element list for the constraint.
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars, obj_coeffs, bounds, A, A_csc, A_csr, [constraints[idx]], rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Partitions the block defined by var_indices and constr_indices based on the
        constraint composition score computed on the corresponding submatrix.
        
        Steps:
        1. Construct sub-arrays for variables, bounds, constraints, and rhs for the block.
        2. Extract the submatrix of A corresponding to these indices.
        3. Call score_constraints on the sub-arrays and submatrix.
        4. Group the original constraint indices by these scores using NumPy.
        5. Group all variable indices together (since this rule does not affect variables).
        6. Sort the constraint groups by their score in descending order.
        7. Form the partition map as the Cartesian product of these groups.
        
        Returns a dictionary mapping block labels to tuples of NumPy arrays:
            { label: (var_indices_array, constr_indices_array) }
        """
        # Ensure var_indices and constr_indices are NumPy arrays.
        var_indices = np.array(var_indices)
        constr_indices = np.array(constr_indices)
        
        # Construct sub-arrays for the current block.
        vars_sub = np.array(vars)[var_indices]
        bounds_sub = np.array(bounds)[var_indices]   # For interface consistency.
        constr_sub = np.array(constraints)[constr_indices]
        rhs_sub = np.array(rhs)[constr_indices] if rhs is not None else None

        # Ensure A is in CSR for efficient row slicing:
        row_slice = A_csr[constr_indices, :]
        submatrix = row_slice.tocsc()[:, var_indices]
        submatrix_csc = submatrix
        submatrix_csr = submatrix.tocsr()
        
        # Compute constraint scores on the submatrix.
        sub_scores = np.array(self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        
        # Group the original constraint indices by their computed scores using NumPy.
        unique_scores = np.unique(sub_scores)
        constr_groups = {}
        for score in unique_scores:
            mask = (sub_scores == score)
            constr_groups[score] = constr_indices[mask]
        
        # All variables are grouped together (since this rule does not differentiate them).
        var_groups = {0: var_indices}
        
        # Sort the unique scores in descending order.
        sorted_scores = np.sort(np.array(list(constr_groups.keys())))[::-1]
        
        # Form the partition map: for each sorted score, assign the entire var_indices and the corresponding constraint group.
        partition_map = {}
        label = 0
        for score in sorted_scores:
            partition_map[label] = (var_indices, constr_groups[score])
            label += 1
                    
        return partition_map

