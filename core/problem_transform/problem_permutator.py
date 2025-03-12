import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import coo_matrix
from utils.logging_handler import LoggingHandler

class ProblemPermutator:
    def __init__(self, gp_env, original_model):
        self.gp_env = gp_env
        self.logger = LoggingHandler().get_logger()
        self.original_model = original_model

    def permutation_matrix(self, perm, sparse=True):
        """
        Build a permutation matrix P from a permutation array perm,
        where perm[i] = j indicates that row i of P has a 1 in column j.
        
        If sparse=True, return a SciPy COO sparse matrix;
        otherwise, return a NumPy dense array.
        """
        n = len(perm)
        row_idx = np.arange(n)
        col_idx = perm
        data = np.ones(n, dtype=np.float64)
        
        if sparse:
            return coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
        else:
            P = np.zeros((n, n))
            P[row_idx, col_idx] = 1
            return P 

    def create_permuted_problem(self, 
                                var_group_size=1, 
                                constr_group_size=1, 
                                var_group_percent=None, 
                                constr_group_percent=None):
        """
        Create a permuted Gurobi model, returning:
          - permuted_model: a new Gurobi model with permuted variables & constraints
          - var_permutation: the permutation array for columns
          - constr_permutation: the permutation array for rows
          - P_col: the column-permutation matrix
          - P_row: the row-permutation matrix

        The full list of indices is always used, but the permutation is done by grouping:
         • Fixed-size groups: set var_group_size or constr_group_size to a desired block size (e.g. 2).
         • Percentage groups: set var_group_percent or constr_group_percent (e.g. 20 for 20% of indices per block);
           this divides the indices into ceil(100/percentage) blocks.
        """
        # --- Determine sizes ---
        num_vars = self.original_model.NumVars
        num_constrs = self.original_model.NumConstrs

        # --- Generate variable permutation ---
        if var_group_percent is not None:
            # Calculate block size so that each block is approximately var_group_percent of the total
            block_size = int(np.ceil(num_vars * (var_group_percent / 100.0)))
            var_permutation = self.group_permutation(num_vars, block_size)
        else:
            var_permutation = self.group_permutation(num_vars, var_group_size)

        # --- Generate constraint permutation ---
        if constr_group_percent is not None:
            block_size = int(np.ceil(num_constrs * (constr_group_percent / 100.0)))
            constr_permutation = self.group_permutation(num_constrs, block_size)
        else:
            constr_permutation = self.group_permutation(num_constrs, constr_group_size)

        # --- Apply permutation using the helper method ---
        permuted_model = self.apply_permutation(self.original_model, var_permutation, constr_permutation)

        # Build permutation matrices (for analysis or reconstruction if needed)
        P_col = self.permutation_matrix(var_permutation, sparse=True)
        P_row = self.permutation_matrix(constr_permutation, sparse=True)
        return permuted_model, var_permutation, constr_permutation, P_col, P_row

    def group_permutation(self, n, group_size):
        """
        Partition n indices into contiguous groups of size 'group_size' 
        (the last group may be smaller) and randomly permute the order of these groups.
        The resulting permutation is the concatenation of the permuted groups.
        """
        indices = np.arange(n)
        # Partition indices into groups
        groups = [list(indices[i: i + group_size]) for i in range(0, n, group_size)]
        # Permute the groups randomly
        np.random.shuffle(groups)
        # Flatten the list of groups back into a single permutation list
        new_perm = [idx for group in groups for idx in group]
        return np.array(new_perm)

    def apply_permutation(self, model, var_permutation, constr_permutation):
        """
        Reorder the given model according to the provided variable and constraint permutations.
        
        Parameters:
            model : gurobipy.Model
                The model to be reordered.
            var_permutation : array-like
                The new ordering for variables (indices referring to model.getVars()).
            constr_permutation : array-like
                The new ordering for constraints (indices referring to model.getConstrs()).
        
        Returns:
            new_model : gurobipy.Model
                A new model with variables and constraints reordered accordingly.
        """
        original_vars = model.getVars()
        constrs = model.getConstrs()
        A = model.getA()  # Original constraint matrix

        num_vars = len(original_vars)
        num_constrs = len(constrs)

        # Create a new model to hold the permuted variables and constraints.
        new_model = gp.Model(env=self.gp_env)

        # --- Reorder variables ---
        new_vars = []
        for new_idx, old_idx in enumerate(var_permutation):
            old_var = original_vars[old_idx]
            new_var = new_model.addVar(
                lb=old_var.LB,
                ub=old_var.UB,
                obj=old_var.Obj,
                vtype=old_var.VType,
                name=f"x{new_idx+1}"
            )
            new_vars.append(new_var)
        new_model.update()

        # --- Reorder constraints ---
        # Convert A to CSC format for easier slicing.
        A_csc = A.tocsc()
        # First, reorder the columns (variables)
        A_tmp = A_csc[:, var_permutation]
        # Then, reorder the rows (constraints)
        A_perm = A_tmp[constr_permutation, :]
        A_perm_csr = A_perm.tocsr()
        indptr = A_perm_csr.indptr
        indices = A_perm_csr.indices
        data = A_perm_csr.data

        for new_row_idx in range(num_constrs):
            old_constr = constrs[constr_permutation[new_row_idx]]
            expr = gp.LinExpr()
            start = indptr[new_row_idx]
            end = indptr[new_row_idx+1]
            for idx in range(start, end):
                col_idx = indices[idx]
                coeff = data[idx]
                expr.add(new_vars[col_idx], float(coeff))
            rhs_value = old_constr.RHS
            if old_constr.Sense == GRB.LESS_EQUAL:
                new_model.addConstr(expr <= rhs_value)
            elif old_constr.Sense == GRB.GREATER_EQUAL:
                new_model.addConstr(expr >= rhs_value)
            else:  # GRB.EQUAL
                new_model.addConstr(expr == rhs_value)

                new_model.ModelSense = model.ModelSense
        new_model.update()
        return new_model

    def permutation_distance(self, row_perm1, col_perm1, row_perm2, col_perm2,
                             row_dist_method="kendall_tau",
                             col_dist_method="kendall_tau",
                             alpha=1.0, beta=1.0):
        """
        Compute a combined distance between two pairs of permutations.
        row_dist_method and col_dist_method can be 'kendall_tau', 'hamming', etc.
        """
        if row_dist_method == "kendall_tau":
            dist_fn_rows = self.kendall_tau_distance
        else:
            dist_fn_rows = self.hamming_distance

        if col_dist_method == "kendall_tau":
            dist_fn_cols = self.kendall_tau_distance
        else:
            dist_fn_cols = self.hamming_distance

        d_rows = dist_fn_rows(row_perm1, row_perm2)
        d_cols = dist_fn_cols(col_perm1, col_perm2)
        return alpha * d_rows + beta * d_cols

    def hamming_distance(self, perm1, perm2):
        if len(perm1) != len(perm2):
            raise ValueError("Permutations must be the same length.")
        return np.sum(np.array(perm1) != np.array(perm2))

    def kendall_tau_distance(self, perm1, perm2):
        n = len(perm1)
        if len(perm2) != n:
            raise ValueError("Permutations must be the same length.")

        # Map elements of perm2 to their positions
        pos_in_perm2 = [0] * n
        for i, val in enumerate(perm2):
            pos_in_perm2[val] = i

        # Transform perm1 into the positions from perm2
        transformed = [pos_in_perm2[val] for val in perm1]

        # Use a merge sort based inversion count
        _, inv_count = self._count_inversions(transformed)
        return inv_count

    def _count_inversions(self, arr):
        # Base case: a single element has zero inversions
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        left, inv_left = self._count_inversions(arr[:mid])
        right, inv_right = self._count_inversions(arr[mid:])
        merged, inv_split = self._merge_count(left, right)
        return merged, inv_left + inv_right + inv_split

    def _merge_count(self, left, right):
        merged = []
        inv_count = 0
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv_count += len(left) - i  # Count inversions: all remaining left items are greater
                j += 1
        # Append remaining elements (no inversions added)
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inv_count
