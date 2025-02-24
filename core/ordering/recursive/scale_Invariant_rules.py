import numpy as np
import math
from collections import defaultdict
from core.ordering.ordering_rule_interface import OrderingRule
from gurobipy import GRB

class ConstraintIntegerCountRule(OrderingRule):
    """
    Counts the number of integer variables (BINARY, INTEGER, or SEMIINT) appearing
    in each constraint.
    """
    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        # This rule does not affect variable scores.
       return np.zeros(len(vars), dtype=int)

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        For each constraint (row) in A, count the number of variables that appear with a
        nonzero coefficient and whose VType is one of GRB.BINARY, GRB.INTEGER, or GRB.SEMIINT.
        """
        num_constraints, num_vars = A.shape

        # Precompute an indicator for integer-like variables.
        # 1 if variable is binary, integer, or semiint; 0 otherwise.
        integer_indicator = np.array(
            [1 if var.VType in [GRB.BINARY, GRB.INTEGER, GRB.SEMIINT] else 0 for var in vars],
            dtype=int
        )

        if hasattr(A, "tocsr"):
            
            # Create a binary indicator matrix from A:
            A_bool = A_csr.copy()
            A_bool.data = np.ones_like(A_bool.data)
            # Multiply each column by the corresponding indicator.
            # The .multiply method applies elementwise multiplication by broadcasting the dense array.
            A_int = A_bool.multiply(integer_indicator)
            # For each row, count nonzero entries.
            scores = A_int.getnnz(axis=1)
        else:
            # Fallback for dense A.
            A_dense = np.array(A)
            A_bool = (np.abs(A_dense) > self.tol).astype(int)
            # Multiply each column by its indicator (broadcasting).
            A_int = A_bool * integer_indicator  # shape (num_constraints, num_vars)
            scores = np.sum(A_int, axis=1)

        return scores

    # --- Methods for Rectangular Block Ordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a tuple containing the score for a single variable.
        (For this rule, variables are not scored, so it is always 0.)
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, A_csc, A_csr, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a tuple containing the score for a single constraint.
        Here the score is the number of integer variables (BINARY, INTEGER, SEMIINT)
        that appear in the constraint.
        """
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars,
                                       obj_coeffs,
                                       bounds,
                                       A,
                                       [constraints[idx]],
                                       rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Partitions the block (defined by var_indices and constr_indices) using the integer count.
        
        Process:
          1. Construct sub-lists for variables, bounds, constraints, and (optionally) rhs.
          2. Extract the submatrix corresponding to these indices.
          3. Compute variable and constraint scores on the submatrix.
          4. Group the original indices by their computed scores.
          5. Form the partition as the Cartesian product of the variable and constraint groups.
        
        Returns a dictionary mapping labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Sub-lists for the block.
        vars_sub = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub = [rhs[i] for i in constr_indices] if rhs is not None else None

        # Extract the submatrix corresponding to the block.
        # Ensure A is in CSR for efficient row slicing:
        
        row_slice = A_csr[constr_indices, :]
        submatrix = row_slice.tocsc()[:, var_indices]
        submatrix_csc = submatrix.tocsc()
        submatrix_csr = submatrix.tocsr()

        # Compute scores on the submatrix.
        sub_var_scores = np.array(self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        sub_constr_scores = np.array(self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        
        # Group the original variable indices by their computed score using vectorized masking.
        unique_var_scores = np.unique(sub_var_scores)
        var_groups = {score: var_indices[sub_var_scores == score] for score in unique_var_scores}

        # Group the original constraint indices by their computed score.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_groups = {score: constr_indices[sub_constr_scores == score] for score in unique_constr_scores}
        
        # Form the partition map as the Cartesian product of the variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v in unique_var_scores:
            for score_c in unique_constr_scores:
                partition_map[label] = (var_groups[score_v], constr_groups[score_c])
                label += 1

        return partition_map


class ConstraintContinuousCountRule(OrderingRule):
    """
    Counts the number of continuous variables (CONTINUOUS or SEMICONT) appearing
    in each constraint.
    """
    def __init__(self):
        self.tol = 1e-15  # tolerance for considering a value nonzero

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        # This rule does not affect variable scores.
       return np.zeros(len(vars), dtype=int)

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        For each constraint (row) in A, count the number of variables that appear with a
        nonzero coefficient and whose VType is either GRB.CONTINUOUS or GRB.SEMICONT.
        
        This version uses vectorized operations:
        - It precomputes an indicator vector for continuous/semi-continuous variables.
        - It converts A to a binary matrix (1 for nonzero entries, 0 otherwise, based on self.tol).
        - It then computes the dot product per row to get the count.
        """
        # Precompute an indicator for continuous/semi-continuous variables.
        is_cont = np.array(
            [1 if var.VType in [GRB.CONTINUOUS, GRB.SEMICONT] else 0 for var in vars],
            dtype=int
        )
        
        tol = self.tol  # tolerance for nonzero

        # Process sparse A if possible.
        if hasattr(A, "tocsr"):
            
            # Create a binary version of A: 1 if |entry| > tol, 0 otherwise.
            binary_data = (np.abs(A_csr.data) > tol).astype(int)
            A_bin = A_csr.copy()
            A_bin.data = binary_data
            # Multiply the binary matrix with the indicator vector.
            scores = A_bin.dot(is_cont)
        else:
            # For dense matrices, convert A to a NumPy array.
            A_dense = np.array(A)
            binary_mask = (np.abs(A_dense) > tol).astype(int)
            scores = np.sum(binary_mask * is_cont, axis=1)
        
        return scores

    # --- Methods for Rectangular Block Ordering ---

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a tuple containing the score for a single variable.
        (For this rule, variables are not scored, so it is always 0.)
        """
        score = self.score_variables([vars[idx]],
                                     obj_coeffs[idx:idx+1],
                                     [bounds[idx]],
                                     A, A_csc, A_csr, constraints, rhs)[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Returns a tuple containing the score for a single constraint.
        Here the score is the number of continuous variables (CONTINUOUS, SEMICONT)
        that appear in the constraint.
        """
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars,
                                       obj_coeffs,
                                       bounds,
                                       A,
                                       [constraints[idx]],
                                       rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        """
        Partitions the block (defined by var_indices and constr_indices) using the continuous count.
        
        Process:
          1. Construct sub-lists for variables, bounds, constraints, and (optionally) rhs.
          2. Extract the submatrix corresponding to these indices.
          3. Compute variable and constraint scores on the submatrix.
          4. Group the original indices by their computed scores.
          5. Form the partition as the Cartesian product of the variable and constraint groups.
        
        Returns a dictionary mapping labels to tuples:
            { label: (list_of_variable_indices, list_of_constraint_indices) }
        """
        # Ensure var_indices and constr_indices are NumPy arrays.
        var_indices = np.array(var_indices)
        constr_indices = np.array(constr_indices)
        # Construct sub-arrays for the current block.
        vars_sub = np.array(vars)[var_indices]
        bounds_sub = np.array(bounds)[var_indices]   # For interface consistency.
        constr_sub = np.array(constraints)[constr_indices]
        rhs_sub = np.array(rhs)[constr_indices] if rhs is not None else None

        # Extract the submatrix corresponding to the current block.
        # Ensure A is in CSR for efficient row slicing:
        
        row_slice = A_csr[constr_indices, :]
        submatrix = row_slice.tocsc()[:, var_indices]
        submatrix_csc = submatrix.tocsc()
        submatrix_csr = submatrix.tocsr()

        # Compute scores on the submatrix.
        sub_var_scores = np.array(self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        sub_constr_scores = np.array(self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        
        # Group the original variable indices by their computed score using vectorized masking.
        unique_var_scores = np.unique(sub_var_scores)
        var_groups = {score: var_indices[sub_var_scores == score] for score in unique_var_scores}

        # Group the original constraint indices by their computed score.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_groups = {score: constr_indices[sub_constr_scores == score] for score in unique_constr_scores}
        
        # Form the partition map as the Cartesian product of the variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v in unique_var_scores:
            for score_c in unique_constr_scores:
                partition_map[label] = (var_groups[score_v], constr_groups[score_c])
                label += 1

        return partition_map
class BothBoundsFiniteCountRule(OrderingRule):
    """
    Counts variables that have both bounds finite.

    For a single variable:
      - Score 1 if both lower and upper bounds are finite.
      - Otherwise, score 0.
    For a constraint, the rule sums the scores (over nonzero coefficients).
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        # Convert bounds to a NumPy array of shape (n_vars, 2).
        bounds_arr = np.array(bounds, dtype=float)
        # Create a boolean mask: True if both bounds are finite.
        finite_mask = ~np.isinf(bounds_arr).any(axis=1)
        # Convert mask to int (1 for finite, 0 for nonfinite) and apply scaling.
        scores = finite_mask.astype(int) * self.scaling
        return scores  # (NumPy array; convert to list if needed)

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Convert bounds to a NumPy array.
        bounds_arr = np.array(bounds, dtype=float)
        # Create an indicator for each variable: 1 if both bounds are finite, 0 otherwise.
        indicator = (~np.isinf(bounds_arr).any(axis=1)).astype(int)  # shape (n_vars,)
        
        if hasattr(A, "tocsr"):
            
            # Create a binary version of A: 1 if coefficient is nonzero, 0 otherwise.
            # (You might add a tolerance if desired; here we treat any nonzero as nonzero.)
            binary = A_csr.copy()
            binary.data = (np.abs(binary.data) > 0).astype(int)
            
            # For each nonzero, multiply by the corresponding indicator from the variable.
            # Get the column indices for each nonzero.
            col_indices = A_csr.indices
            binary.data = binary.data * indicator[col_indices]
            
            # The score for each constraint (row) is then the sum of the binary data in that row.
            scores = binary.sum(axis=1).A1  # .A1 converts the result to a flat NumPy array.
        else:
            # Fallback for dense A.
            A_dense = np.array(A)
            # Create a binary mask for nonzero entries.
            binary = (np.abs(A_dense) > 0).astype(int)
            # Multiply each column by the indicator.
            binary = binary * indicator[np.newaxis, :]
            scores = np.sum(binary, axis=1)
        
        return scores  # (NumPy array; convert to list if necessary)

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        score = self.score_variables(
                    [vars[idx]],
                    obj_coeffs[idx:idx+1],
                    [bounds[idx]],
                    A, A_csc, A_csr, constraints, rhs
                )[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars,
                                       obj_coeffs,
                                       bounds,
                                       A,
                                       [constraints[idx]],
                                       rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Ensure var_indices and constr_indices are NumPy arrays.
        var_indices = np.array(var_indices)
        constr_indices = np.array(constr_indices)
        # Construct sub-arrays for the current block.
        vars_sub = np.array(vars)[var_indices]
        bounds_sub = np.array(bounds)[var_indices]   # For interface consistency.
        constr_sub = np.array(constraints)[constr_indices]
        rhs_sub = np.array(rhs)[constr_indices] if rhs is not None else None

        # Extract the submatrix.
        # Ensure A is in CSR for efficient row slicing:
        
        row_slice = A_csr[constr_indices, :]
        submatrix = row_slice.tocsc()[:, var_indices]
        submatrix_csc = submatrix.tocsc()
        submatrix_csr = submatrix.tocsr()

        # Compute scores on the submatrix.
        sub_var_scores = np.array(self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        sub_constr_scores = np.array(self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        
        # Group the original variable indices by their computed score using vectorized masking.
        unique_var_scores = np.unique(sub_var_scores)
        var_groups = {score: var_indices[sub_var_scores == score] for score in unique_var_scores}

        # Group the original constraint indices by their computed score.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_groups = {score: constr_indices[sub_constr_scores == score] for score in unique_constr_scores}
        
        # Form the partition map as the Cartesian product of the variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v in unique_var_scores:
            for score_c in unique_constr_scores:
                partition_map[label] = (var_groups[score_v], constr_groups[score_c])
                label += 1

        return partition_map
class OneBoundFiniteCountRule(OrderingRule):
    """
    Counts variables that have exactly one finite bound.
    
    For a single variable:
      - Score 1 if exactly one of the bounds is finite.
      - Otherwise, score 0.
    For a constraint, the rule sums the scores over variables with nonzero coefficients.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        # Convert bounds to a NumPy array of shape (n_vars, 2)
        bounds_arr = np.array(bounds, dtype=float)
        # A finite bound: not infinite.
        finite = ~np.isinf(bounds_arr)
        # Count finite bounds per variable (0, 1, or 2).
        finite_count = np.sum(finite, axis=1)
        # Score = 1 if exactly one bound is finite, 0 otherwise, then scaled.
        scores = (finite_count == 1).astype(int) * self.scaling
        return scores  # (If needed, use .tolist())

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Convert bounds to a NumPy array.
        bounds_arr = np.array(bounds, dtype=float)
        # Indicator: 1 for a variable that has exactly one finite bound, 0 otherwise.
        finite = ~np.isinf(bounds_arr)
        finite_count = np.sum(finite, axis=1)
        indicator = (finite_count == 1).astype(int)  # shape (n_vars,)
        
        # We want to count, for each constraint (row), the number of variables with
        # nonzero coefficients and with indicator==1.
        if hasattr(A, "tocsr"):
            
            # Convert A to a binary matrix: set nonzero entries (ignoring sign) to 1.
            A_bin = A_csr.copy()
            A_bin.data = (np.abs(A_bin.data) != 0).astype(int)
            # Multiply each column by the indicator.
            # Using sparse dot-product: For each row, the dot product with 'indicator' gives
            # the sum over columns of (nonzero_flag * indicator).
            scores = A_bin.dot(indicator) * self.scaling
        else:
            # Dense fallback: assume A is a NumPy array.
            A_dense = np.array(A)
            binary_mask = (np.abs(A_dense) != 0).astype(int)  # shape (n_constraints, n_vars)
            scores = np.sum(binary_mask * indicator[np.newaxis, :], axis=1) * self.scaling
        
        return scores  # (If needed, use .tolist())

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        score = self.score_variables(
                    [vars[idx]],
                    obj_coeffs[idx:idx+1],
                    [bounds[idx]],
                    A, A_csc, A_csr, constraints, rhs
                )[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars,
                                       obj_coeffs,
                                       bounds,
                                       A,
                                       [constraints[idx]],
                                       rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        vars_sub   = [vars[i] for i in var_indices]
        bounds_sub = [bounds[i] for i in var_indices]
        constr_sub = [constraints[i] for i in constr_indices]
        rhs_sub    = [rhs[i] for i in constr_indices] if rhs is not None else None

        # Ensure A is in CSR for efficient row slicing:
        
        row_slice = A_csr[constr_indices, :]
        submatrix = row_slice.tocsc()[:, var_indices]
        submatrix_csc = submatrix.tocsc()
        submatrix_csr = submatrix.tocsr()

        sub_var_scores    = self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub)
        sub_constr_scores = self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub)

        var_groups = defaultdict(list)
        for idx, score in zip(var_indices, sub_var_scores):
            var_groups[score].append(idx)
        constr_groups = defaultdict(list)
        for idx, score in zip(constr_indices, sub_constr_scores):
            constr_groups[score].append(idx)

        partition_map = {}
        label = 0
        for score_v, vgroup in var_groups.items():
            for score_c, cgroup in constr_groups.items():
                partition_map[label] = (vgroup, cgroup)
                label += 1
        return partition_map
class BothBoundsInfiniteCountRule(OrderingRule):
    """
    Counts variables that have both bounds infinite.
    
    For a single variable:
      - Score 1 if both lower and upper bounds are infinite.
      - Otherwise, score 0.
    For a constraint, the rule sums the scores over variables with nonzero coefficients.
    """
    def __init__(self, scaling=1):
        self.scaling = scaling

    def score_variables(self, vars, obj_coeffs, bounds, A, A_csc, A_csr,  constraints, rhs):
        # Convert bounds to a NumPy array with shape (n_vars, 2)
        bounds_arr = np.array(bounds, dtype=float)
        # A variable has both bounds infinite if both entries are infinite.
        mask = np.all(np.isinf(bounds_arr), axis=1)  # Boolean array: True if both are inf.
        scores = mask.astype(int) * self.scaling
        return scores  # Return as a NumPy array; convert to list if needed

    def score_constraints(self, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        # Convert bounds to a NumPy array.
        bounds_arr = np.array(bounds, dtype=float)
        # Create an indicator for variables that have both bounds infinite.
        indicator = np.all(np.isinf(bounds_arr), axis=1).astype(int)  # shape (n_vars,)
        
        # Process A as a sparse matrix if possible.
        if hasattr(A, "tocsr"):
            
            # Create a binary version of A: 1 for nonzero entries.
            A_bin = A_csr.copy()
            A_bin.data = (np.abs(A_bin.data) != 0).astype(int)
            # For each nonzero, multiply by the indicator corresponding to its column.
            A_bin.data = A_bin.data * indicator[A_csr.indices]
            # Sum along rows: each row's sum gives the count of nonzero entries coming from variables with both bounds infinite.
            scores = A_bin.sum(axis=1).A1 * self.scaling
        else:
            # Fallback for dense matrices.
            A_dense = np.array(A)
            binary = (np.abs(A_dense) != 0).astype(int)  # shape (n_constraints, n_vars)
            scores = np.sum(binary * indicator[np.newaxis, :], axis=1) * self.scaling
        
        return scores  # Return as a NumPy array; convert to list if needed

    def score_matrix_for_variable(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        score = self.score_variables(
                    [vars[idx]],
                    obj_coeffs[idx:idx+1],
                    [bounds[idx]],
                    A, A_csc, A_csr, constraints, rhs
                )[0]
        return (score,)

    def score_matrix_for_constraint(self, idx, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
        rhs_single = [rhs[idx]] if rhs is not None else None
        score = self.score_constraints(vars,
                                       obj_coeffs,
                                       bounds,
                                       A,
                                       [constraints[idx]],
                                       rhs_single)[0]
        return (score,)

    def score_matrix(self, var_indices, constr_indices, vars, obj_coeffs, bounds, A, A_csc, A_csr, constraints, rhs):
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
        submatrix_csc = submatrix.tocsc()
        submatrix_csr = submatrix.tocsr()

        # Compute scores on the submatrix.
        sub_var_scores = np.array(self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        sub_constr_scores = np.array(self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, submatrix_csc, submatrix_csr, constr_sub, rhs_sub))
        
        # Group the original variable indices by their computed score using vectorized masking.
        unique_var_scores = np.unique(sub_var_scores)
        var_groups = {score: var_indices[sub_var_scores == score] for score in unique_var_scores}

        # Group the original constraint indices by their computed score.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_groups = {score: constr_indices[sub_constr_scores == score] for score in unique_constr_scores}
        
        # Form the partition map as the Cartesian product of the variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v in unique_var_scores:
            for score_c in unique_constr_scores:
                partition_map[label] = (var_groups[score_v], constr_groups[score_c])
                label += 1

        return partition_map
