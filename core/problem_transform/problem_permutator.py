import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import coo_matrix
from utils.logging_handler import LoggingHandler
from core.problem_transform.distance import DistanceMetric

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
        var_permutation = self.group_permutation(num_vars, var_group_size, rng)
        constr_permutation = self.group_permutation(num_constrs, constr_group_size, rng)

        # Apply the permutations to create the new model.
        permuted_model = self.apply_permutation(self.original_model, var_permutation, constr_permutation)

        # Build permutation matrices for potential further analysis.
        P_col = self.permutation_matrix(var_permutation, sparse=True)
        P_row = self.permutation_matrix(constr_permutation, sparse=True)
        return permuted_model, var_permutation, constr_permutation, P_col, P_row

    def group_permutation(self, n, group_size, rng):
        """
        Partition n indices into contiguous groups of size 'group_size' 
        (the last group may be smaller) and randomly permute the order of these groups.
        The resulting permutation is the concatenation of the permuted groups.
        """
        indices = np.arange(n)
        # Partition indices into groups
        groups = [list(indices[i: i + group_size]) for i in range(0, n, group_size)]
        # Permute the groups randomly using the provided random generator
        rng.shuffle(groups)
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

    def permutation_distance(self, row_perm1, col_perm1, row_perm2, col_perm2, row_metric: DistanceMetric, col_metric: DistanceMetric,
                         alpha=1.0, beta=1.0):
        """
        Compute the overall distance as a weighted sum of row and column distances.
        If column permutations are not provided, only row distance is computed.
        """
        d_rows = row_metric.compute(row_perm1, row_perm2)
        d_cols = col_metric.compute(col_perm1, col_perm2)
        return alpha * d_rows + beta * d_cols


    # --------------------------------------------------------------------------
    # ------------------ NEW METHODS: BUILD ADJACENCY + DISTANCE ---------------
    # --------------------------------------------------------------------------
    def get_variable_adjacency(self, model):
        """
        Build an adjacency dictionary for variables.
        Two variables are considered adjacent if they appear together in at least one constraint.
        
        Returns:
            adjacency: dict
                For each variable index j, adjacency[j] is a set of variable indices that
                appear in at least one common constraint with j.
        """
        A = model.getA().tocoo()  # Get constraint matrix in COO format
        num_vars = model.NumVars

        # Build a mapping from each constraint to the set of variables in that constraint.
        constr_vars = {}
        for row, col, value in zip(A.row, A.col, A.data):
            if value != 0:
                constr_vars.setdefault(row, set()).add(col)
        
        # Build variable adjacency: two variables are adjacent if they appear in the same constraint.
        adjacency = {j: set() for j in range(num_vars)}
        for var_set in constr_vars.values():
            for j in var_set:
                for k in var_set:
                    if j != k:
                        adjacency[j].add(k)
        return adjacency
    
    def get_constraint_adjacency(self, model):
        """
        Build an adjacency dictionary for constraints.
        Two constraints are considered adjacent if they share at least one variable.
        
        Returns:
            adjacency: dict
                adjacency[i] is a set of constraint indices that share >=1 var with constraint i
        """
        A = model.getA().tocoo()  # get constraint matrix in COO format
        num_constrs = model.NumConstrs
        num_vars = model.NumVars

        # 1) For each constraint i, record which variables appear
        constr_vars = {i: set() for i in range(num_constrs)}
        for row_idx, col_idx, value in zip(A.row, A.col, A.data):
            if value != 0:
                constr_vars[row_idx].add(col_idx)

        # 2) Build adjacency by union over shared variables
        adjacency = {i: set() for i in range(num_constrs)}
        # For each variable, we gather all constraints it appears in, then mark them adjacent.
        var_to_constrs = {}
        for c_idx in range(num_constrs):
            for v_idx in constr_vars[c_idx]:
                var_to_constrs.setdefault(v_idx, []).append(c_idx)

        # Now fill adjacency
        for v, c_list in var_to_constrs.items():
            for c1 in c_list:
                for c2 in c_list:
                    if c1 != c2:
                        adjacency[c1].add(c2)

        return adjacency

    
