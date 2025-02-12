# core/problem_scaler.py

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import coo_matrix  # for optional output of diagonal scaling matrices
from utils.logging_handler import LoggingHandler

class ProblemScaler:
    def __init__(self, gp_env, original_model):
        """
        Initialize with a Gurobi environment and an already built original model.
        """
        self.gp_env = gp_env
        self.logger = LoggingHandler().get_logger()
        self.original_model = original_model

    def create_scaled_problem_with_scales(self, row_scales, col_scales):
        """
        Create a scaled version of the original model using provided scaling factors.

        Parameters
        ----------
        row_scales : array-like of floats, shape (num_constrs,)
            Scaling factors for each constraint (row). Must be nonzero.
        col_scales : array-like of floats, shape (num_vars,)
            Scaling factors for each variable (column). Must be nonzero.

        Returns
        -------
        scaled_model : gurobipy.Model
            The new scaled model.
        row_scales : np.ndarray
            The row scaling factors used.
        col_scales : np.ndarray
            The column scaling factors used.
        D_row : scipy.sparse.coo_matrix
            The (diagonal) row scaling matrix.
        D_col : scipy.sparse.coo_matrix
            The (diagonal) column scaling matrix.
        """
        row_scales = np.array(row_scales)
        col_scales = np.array(col_scales)

        self._validate_scales(row_scales, col_scales)
        return self._build_scaled_problem(row_scales, col_scales)

    def create_scaled_problem_random(self,
                                     n_rows_to_scale=0,
                                     n_cols_to_scale=0,
                                     row_scale_bounds=(0.5, 2.0),
                                     col_scale_bounds=(0.5, 2.0),
                                     row_allow_negative=True,
                                     col_allow_negative=False):
        """
        Create a scaled version of the original model by randomly choosing a subset
        of rows and columns to scale. Rows or columns not chosen will have a scaling factor of 1.

        Parameters
        ----------
        n_rows_to_scale : int, default 0
            Number of constraints (rows) to scale. The rest will be scaled by 1.
        n_cols_to_scale : int, default 0
            Number of variables (columns) to scale. The rest will be scaled by 1.
        row_scale_bounds : tuple (low, high), default (0.5, 2.0)
            Bounds for drawing a random row scaling factor (absolute value).
        col_scale_bounds : tuple (low, high), default (0.5, 2.0)
            Bounds for drawing a random column scaling factor (absolute value).
        row_allow_negative : bool, default True
            If True, each chosen row scaling factor has a 50% chance to be negative.
        col_allow_negative : bool, default False
            If True, each chosen column scaling factor has a 50% chance to be negative.

        Returns
        -------
        scaled_model : gurobipy.Model
            The new scaled model.
        row_scales : np.ndarray
            The row scaling factors used.
        col_scales : np.ndarray
            The column scaling factors used.
        D_row : scipy.sparse.coo_matrix
            The (diagonal) row scaling matrix.
        D_col : scipy.sparse.coo_matrix
            The (diagonal) column scaling matrix.
        """
        # Get numbers of variables and constraints.
        num_constrs = len(self.original_model.getConstrs())
        num_vars = len(self.original_model.getVars())

        # Initialize scaling factors with no scaling (i.e., factor of 1)
        row_scales = np.ones(num_constrs)
        col_scales = np.ones(num_vars)

        # Randomly select rows to scale, if requested.
        if n_rows_to_scale > 0:
            chosen_rows = np.random.choice(num_constrs, n_rows_to_scale, replace=False)
            for i in chosen_rows:
                scale_val = np.random.uniform(*row_scale_bounds)
                if row_allow_negative and np.random.rand() < 0.5:
                    scale_val *= -1
                row_scales[i] = scale_val

        # Randomly select columns to scale, if requested.
        if n_cols_to_scale > 0:
            chosen_cols = np.random.choice(num_vars, n_cols_to_scale, replace=False)
            for j in chosen_cols:
                scale_val = np.random.uniform(*col_scale_bounds)
                if col_allow_negative and np.random.rand() < 0.5:
                    scale_val *= -1
                col_scales[j] = scale_val

        self._validate_scales(row_scales, col_scales)
        return self._build_scaled_problem(row_scales, col_scales)

    def _validate_scales(self, row_scales, col_scales):
        """
        Validate that the scaling vectors are of the proper length and nonzero.

        Raises
        ------
        ValueError if the lengths do not match the model or any scale is (close to) zero.
        """
        num_constrs = len(self.original_model.getConstrs())
        num_vars = len(self.original_model.getVars())

        if len(row_scales) != num_constrs:
            raise ValueError("Length of provided row_scales must equal the number of constraints.")
        if len(col_scales) != num_vars:
            raise ValueError("Length of provided col_scales must equal the number of variables.")
        if np.any(np.isclose(row_scales, 0)):
            raise ValueError("All row scaling factors must be nonzero.")
        if np.any(np.isclose(col_scales, 0)):
            raise ValueError("All column scaling factors must be nonzero.")

    def _build_scaled_problem(self, row_scales, col_scales):
        """
        Builds the scaled model with consistent scaling between objective and constraints.
        
        Using the transformation x_j = y_j/s_j (where s_j is col_scales[j]):
        - Variables: x_j = y_j/s_j
        - Constraints: r_i * (sum a_ij * x_j) = r_i * b_i
            becomes: r_i * (sum a_ij * (y_j/s_j)) = r_i * b_i
        - For objective: min sum(c_j * x_j)
            becomes: min sum(c_j * (y_j/s_j))
        """
        original_vars = self.original_model.getVars()
        constrs = self.original_model.getConstrs()
        A = self.original_model.getA()
        
        num_vars = len(original_vars)
        num_constrs = len(constrs)
        
        # Create new model
        scaled_model = gp.Model(env=self.gp_env)
        
        # --- 1) Create scaled variables ---
        new_vars = []
        for j in range(num_vars):
            s = col_scales[j]
            old_var = original_vars[j]
            
            # Scale bounds for x_j = y_j/s_j transformation
            # y_j = s_j * x_j, so bounds are multiplied by s_j
            if s > 0:
                new_lb = s * old_var.LB if old_var.LB != float('-inf') else float('-inf')
                new_ub = s * old_var.UB if old_var.UB != float('inf') else float('inf')
            else:
                new_lb = s * old_var.UB if old_var.UB != float('inf') else float('-inf')
                new_ub = s * old_var.LB if old_var.LB != float('-inf') else float('inf')
            
            # Scale objective: c_j * x_j = c_j * (y_j/s_j) = (c_j/s_j) * y_j
            new_obj = old_var.Obj * s
            
            # Handle variable types
            old_vtype = old_var.VType
            new_vtype = old_vtype
            if old_vtype == GRB.BINARY:
                if not np.isclose(abs(s), 1.0):
                    new_vtype = GRB.INTEGER if (isinstance(new_lb, int) and isinstance(new_ub, int)) else GRB.CONTINUOUS
                    self.logger.warning(f"Binary variable {old_var.VarName} scaled by {s}. Changed to {new_vtype}.")
            elif old_vtype == GRB.INTEGER:
                if not np.isclose(s, round(s)):
                    new_vtype = GRB.CONTINUOUS
                    self.logger.warning(f"Integer variable {old_var.VarName} scaled by non-integer {s}. Changed to continuous.")
            
            new_var = scaled_model.addVar(
                lb=new_lb,
                ub=new_ub,
                obj=new_obj,
                vtype=new_vtype,
                name=f"y{j+1}"
            )
            new_vars.append(new_var)
        
        # --- 2) Create scaled constraints ---
        A_csr = A.tocsr()
        for i in range(num_constrs):
            r = row_scales[i]
            old_constr = constrs[i]
            
            # Scale RHS: r_i * b_i
            new_rhs = r * old_constr.RHS
            
            # Handle constraint sense
            old_sense = old_constr.Sense
            new_sense = old_sense
            if old_sense != GRB.EQUAL:
                new_sense = old_sense if r > 0 else (GRB.GREATER_EQUAL if old_sense == GRB.LESS_EQUAL else GRB.LESS_EQUAL)
            
            # Build constraint expression
            expr = gp.LinExpr()
            row = A_csr.getrow(i)
            for j, a_ij in zip(row.indices, row.data):
                s = col_scales[j]
                # New coefficient: r_i * a_ij/s_j
                # This comes from r_i * (a_ij * (y_j/s_j))
                new_coeff = r * (a_ij / s)
                expr.add(new_vars[j], new_coeff)
            
            # Add constraint with appropriate sense
            if new_sense == GRB.LESS_EQUAL:
                scaled_model.addConstr(expr <= new_rhs, name=f"c{i+1}")
            elif new_sense == GRB.GREATER_EQUAL:
                scaled_model.addConstr(expr >= new_rhs, name=f"c{i+1}")
            else:  # GRB.EQUAL
                scaled_model.addConstr(expr == new_rhs, name=f"c{i+1}")
        
        # Preserve optimization direction
        scaled_model.ModelSense = self.original_model.ModelSense
        scaled_model.update()
        
        # Create diagonal scaling matrices for reference
        D_row = coo_matrix((row_scales, (range(num_constrs), range(num_constrs))))
        D_col = coo_matrix((col_scales, (range(num_vars), range(num_vars))))
        
        return scaled_model, row_scales, col_scales, D_row, D_col
