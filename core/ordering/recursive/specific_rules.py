from core.ordering.ordering_rule_interface import OrderingRule
import numpy as np
from collections import defaultdict

class AllCoefficientsOneRule(OrderingRule):
    """
    Scores variables and constraints based on all non-zero coefficients being 1.
    - Variables: Score is scaling if all non-zero coefficients in the column are 1.
    - Constraints: Score is scaling if all non-zero coefficients in the row are 1.
    """
    
    def __init__(self, scaling=1, tol=1e-12):
        self.scaling = scaling
        self.tol = tol

    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_vars = A.shape[1]
        scores = np.zeros(num_vars, dtype=int)
        
        if hasattr(A, "tocsc"):
            A_csc = A.tocsc()
            # Get column pointers and compute number of nonzeros per column.
            indptr = A_csc.indptr
            counts = np.diff(indptr)  # length = num_vars
            # Compute absolute deviation from 1 for all nonzero elements.
            diff = np.abs(A_csc.data - 1.0)
            # Preallocate an array for the maximum deviation in each column.
            max_diff = np.empty(num_vars, dtype=float)
            # Create a mask for non-empty columns.
            nonempty_mask = counts > 0
            nonempty_idx = np.where(nonempty_mask)[0]
            # Loop over non-empty columns to compute maximum deviation.
            for j in nonempty_idx:
                max_diff[j] = np.max(diff[indptr[j]:indptr[j+1]])
            # For empty columns, set the maximum deviation to np.inf.
            max_diff[~nonempty_mask] = np.inf
                
            # A column scores 'scaling' if its maximum deviation is within tolerance.
            scores = np.where(max_diff <= self.tol, self.scaling, 0)
        else:
            # Fallback using a Python loop if A does not support toscsc.
            for j in range(num_vars):
                column = A[:, j].flatten()
                non_zero = np.abs(column) > self.tol
                non_zero_data = column[non_zero]
                if non_zero_data.size > 0 and np.allclose(non_zero_data, 1.0, atol=self.tol):
                    scores[j] = self.scaling
        return scores


    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        num_constraints = A.shape[0]
        scores = np.zeros(num_constraints, dtype=int)
        
        if hasattr(A, "tocsr"):
            A_csr = A.tocsr()
            indptr = A_csr.indptr
            counts = np.diff(indptr)
            diff = np.abs(A_csr.data - 1.0)
            max_diff = np.empty(num_constraints, dtype=float)
            nonempty_mask = counts > 0
            nonempty_idx = np.where(nonempty_mask)[0]
            for i in nonempty_idx:
                max_diff[i] = np.max(diff[indptr[i]:indptr[i+1]])
            max_diff[~nonempty_mask] = np.inf
            scores = np.where(max_diff <= self.tol, self.scaling, 0)
        else:
            for i in range(num_constraints):
                row = A[i, :].flatten()
                non_zero = np.abs(row) > self.tol
                non_zero_data = row[non_zero]
                if non_zero_data.size > 0 and np.allclose(non_zero_data, 1.0, atol=self.tol):
                    scores[i] = self.scaling
        return scores


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
        # Ensure var_indices and constr_indices are NumPy arrays.
        var_indices = np.array(var_indices)
        constr_indices = np.array(constr_indices)
        # Construct sub-arrays for the current block.
        vars_sub = np.array(vars)[var_indices]
        bounds_sub = np.array(bounds)[var_indices]   # For interface consistency.
        constr_sub = np.array(constraints)[constr_indices]
        rhs_sub = np.array(rhs)[constr_indices] if rhs is not None else None
        
        # Ensure A is in CSR for efficient row slicing:
        A_csr = A.tocsr()
        row_slice = A_csr[constr_indices, :]
        submatrix = row_slice.tocsc()[:, var_indices]
        
        # Compute scores on the sub-block.
        sub_var_scores = np.array(self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub))
        sub_constr_scores = np.array(self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub))
        
        # Ensure original indices are numpy arrays.
        var_indices = np.array(var_indices)
        constr_indices = np.array(constr_indices)
        
        # Group variable indices by their score.
        unique_var_scores = np.unique(sub_var_scores)
        var_partitions = {}
        for score in unique_var_scores:
            mask = (sub_var_scores == score)
            var_partitions[score] = var_indices[mask]
        
        # Group constraint indices by their score.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_partitions = {}
        for score in unique_constr_scores:
            mask = (sub_constr_scores == score)
            constr_partitions[score] = constr_indices[mask]
        
        # Form the partition map as the Cartesian product of variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v, var_group in var_partitions.items():
            for score_c, constr_group in constr_partitions.items():
                partition_map[label] = (var_group, constr_group)
                label += 1
        return partition_map
    

class AllBinaryVariablesRule(OrderingRule):
    """
    Scores variables and constraints based on binary variables.
    - Variables: Score is scaling if the variable is binary (bounds [0,1]).
    - Constraints: Score is scaling if all variables in the constraint are binary.
    """
    
    def __init__(self, scaling=1, tol=1e-12):
        self.scaling = scaling
        self.tol = tol


    def score_variables(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # Convert bounds to a NumPy array (shape: [num_vars, 2])
        bounds_arr = np.array(bounds, dtype=float)
        # Check if lower bound is close to 0 and upper bound close to 1
        lb_close = np.isclose(bounds_arr[:, 0], 0.0, atol=self.tol)
        ub_close = np.isclose(bounds_arr[:, 1], 1.0, atol=self.tol)
        is_binary = lb_close & ub_close
        # Score is scaling if binary, 0 otherwise.
        scores = np.where(is_binary, self.scaling, 0)
        return scores

    def score_constraints(self, vars, obj_coeffs, bounds, A, constraints, rhs):
        # Number of constraints
        num_constraints = A.shape[0]
        # Convert bounds to a NumPy array (shape: [num_vars, 2])
        bounds_arr = np.array(bounds, dtype=float)
        # Precompute a boolean array for binary variables.
        binary_vars = (np.isclose(bounds_arr[:, 0], 0.0, atol=self.tol) &
                       np.isclose(bounds_arr[:, 1], 1.0, atol=self.tol))
        
        scores = np.zeros(num_constraints, dtype=int)
        
        if hasattr(A, "tocsr"):
            A_csr = A.tocsr()
            for i in range(num_constraints):
                start = A_csr.indptr[i]
                end = A_csr.indptr[i+1]
                # Get the indices of nonzero entries in row i.
                row_indices = A_csr.indices[start:end]
                if row_indices.size > 0 and np.all(binary_vars[row_indices]):
                    scores[i] = self.scaling
        else:
            # Fallback for dense or other formats.
            for i in range(num_constraints):
                row = A[i, :].flatten()
                non_zero = np.abs(row) > self.tol
                non_zero_indices = np.where(non_zero)[0]
                if non_zero_indices.size > 0 and np.all(binary_vars[non_zero_indices]):
                    scores[i] = self.scaling
        return scores

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
        # Ensure var_indices and constr_indices are NumPy arrays.
        var_indices = np.array(var_indices)
        constr_indices = np.array(constr_indices)
        # Construct sub-arrays for the current block.
        vars_sub = np.array(vars)[var_indices]
        bounds_sub = np.array(bounds)[var_indices]   # For interface consistency.
        constr_sub = np.array(constraints)[constr_indices]
        rhs_sub = np.array(rhs)[constr_indices] if rhs is not None else None
        
        # Ensure A is in CSR for efficient row slicing:
        A_csr = A.tocsr()
        row_slice = A_csr[constr_indices, :]
        submatrix = row_slice.tocsc()[:, var_indices]
        
        # Compute scores on the sub-block.
        sub_var_scores = np.array(self.score_variables(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub))
        sub_constr_scores = np.array(self.score_constraints(vars_sub, obj_coeffs, bounds_sub, submatrix, constr_sub, rhs_sub))
        
        # Group the original variable indices by their computed score using NumPy.
        unique_var_scores = np.unique(sub_var_scores)
        var_partitions = {}
        for score in unique_var_scores:
            mask = (sub_var_scores == score)
            var_partitions[score] = var_indices[mask]
        
        # Similarly group the constraint indices.
        unique_constr_scores = np.unique(sub_constr_scores)
        constr_partitions = {}
        for score in unique_constr_scores:
            mask = (sub_constr_scores == score)
            constr_partitions[score] = constr_indices[mask]
        
        # Form the partition map as the Cartesian product of variable and constraint groups.
        partition_map = {}
        label = 0
        for score_v in unique_var_scores:
            for score_c in unique_constr_scores:
                partition_map[label] = (var_partitions[score_v], constr_partitions[score_c])
                label += 1
                    
        return partition_map