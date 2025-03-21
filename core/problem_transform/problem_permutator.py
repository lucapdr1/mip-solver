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

    def create_permuted_problem(self, num_subblocks, seed=None):
        """
        Create a permuted Gurobi model where both variables and constraints 
        are partitioned into groups that are then randomly shuffled.
        
        Parameters:
            num_subblocks : int or str
                - If an integer, it is used to determine the number of subblocks for both variables and constraints.
                  For each dimension, if the provided number is greater than or equal to the number of elements,
                  it falls back to the special case where each element is individually permuted.
                - If a string (e.g., "full"), then every variable and constraint is individually permuted.
            seed : int, optional
                The random seed used to generate the permutation. If provided, the permutation
                will be reproducible.
                
        Returns:
            permuted_model : A new Gurobi model with permuted variables and constraints.
            var_permutation : The permutation array for variables.
            constr_permutation : The permutation array for constraints.
            P_col : The column permutation matrix.
            P_row : The row permutation matrix.
        """
        num_vars = self.original_model.NumVars
        num_constrs = self.original_model.NumConstrs

        # Log the seed if provided
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # First, convert num_subblocks if it's a string.
        if isinstance(num_subblocks, str):
            try:
                num_subblocks = int(num_subblocks)
            except ValueError:
                if num_subblocks.lower() in ["full", "all", "individual"]:
                    # Special case: full permutation
                    var_group_size = 1
                    constr_group_size = 1
                else:
                    raise ValueError("Invalid string value for num_subblocks. Use 'full' for full permutation or provide a numeric value.")

        # Now, if num_subblocks is an int, compute the group sizes.
        if isinstance(num_subblocks, int):
            if num_subblocks >= num_vars:
                var_group_size = 1
            else:
                var_group_size = int(np.ceil(num_vars / num_subblocks))
            if num_subblocks >= num_constrs:
                constr_group_size = 1
            else:
                constr_group_size = int(np.ceil(num_constrs / num_subblocks))

        # Generate permutations by grouping indices into blocks of the computed size.
<<<<<<< HEAD
        var_permutation = self.group_permutation(num_vars, var_group_size)
        constr_permutation = self.group_permutation(num_constrs, constr_group_size)
=======
        var_permutation = self.group_permutation(num_vars, var_group_size, rng)
        constr_permutation = self.group_permutation(num_constrs, constr_group_size, rng)
>>>>>>> 5b64d786ac78b9e62ae6bc223ef4619e71bf5428

        # Apply the permutations to create the new model.
        permuted_model = self.apply_permutation(self.original_model, var_permutation, constr_permutation)

        # Build permutation matrices for potential further analysis.
        P_col = self.permutation_matrix(var_permutation, sparse=True)
        P_row = self.permutation_matrix(constr_permutation, sparse=True)
        return permuted_model, var_permutation, constr_permutation, P_col, P_row

<<<<<<< HEAD
    def group_permutation(self, n, group_size):
=======
    def group_permutation(self, n, group_size, rng):
>>>>>>> 5b64d786ac78b9e62ae6bc223ef4619e71bf5428
        """
        Partition n indices into contiguous groups of size 'group_size' 
        (the last group may be smaller) and randomly permute the order of these groups.
        The resulting permutation is the concatenation of the permuted groups.
        """
        indices = np.arange(n)
        # Partition indices into groups
        groups = [list(indices[i: i + group_size]) for i in range(0, n, group_size)]
<<<<<<< HEAD
        # Permute the groups randomly
        np.random.shuffle(groups)
=======
        # Permute the groups randomly using the provided random generator
        rng.shuffle(groups)
>>>>>>> 5b64d786ac78b9e62ae6bc223ef4619e71bf5428
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
