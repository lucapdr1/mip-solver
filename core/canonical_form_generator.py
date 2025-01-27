import gurobipy as gp
from gurobipy import GRB
from core.logging_handler import LoggingHandler
from core.problem_printer import ProblemPrinter
import numpy as np
import hashlib
import scipy.sparse as sp
from networkx import Graph, community

class CanonicalFormGenerator:
    def __init__(self, model):
        self.original_model = model
        self.model = model.copy()
        self.logger = LoggingHandler().get_logger()
        self._initialize_structures()
        
    def _initialize_structures(self):
        """Extract model components with integrity checks"""
        self.vars = self.model.getVars()
        self.constrs = self.model.getConstrs()
        self.A = self.model.getA()  # Constraint matrix in sparse format

        if self.A is None or self.A.shape[0] == 0 or self.A.shape[1] == 0:
            raise ValueError("Constraint matrix is empty or invalid")

        # Validation checks
        if len(self.constrs) != self.A.shape[0]:
            raise ValueError("Mismatch between constraints and matrix rows")

        if len(self.vars) != self.A.shape[1]:
            raise ValueError("Mismatch between variables and matrix columns")

        self.obj_coeffs = np.array([var.Obj for var in self.vars])
        self.rhs = np.array([c.RHS for c in self.constrs])
        self.sense = self.model.ModelSense
    
    def normalize_matrix_consistently(self,A, obj_coeffs, rhs):
        """
        Consistently normalize both rows and columns of matrix A.
        
        Args:
            A (scipy.sparse.csr_matrix): Constraint matrix.
            obj_coeffs (np.ndarray): Objective coefficients.
            rhs (np.ndarray): Right-hand side values.

        Returns:
            tuple: Normalized (A, obj_coeffs, rhs) and the scaling factors.
        """
        A_csr = A.tocsr()

        # Compute row and column norms
        row_norms = np.sqrt(A_csr.power(2).sum(axis=1).A.flatten())  # Row norms
        col_norms = np.sqrt(A_csr.power(2).sum(axis=0).A.flatten())  # Column norms

        # Avoid division by zero
        row_norms[row_norms == 0] = 1
        col_norms[col_norms == 0] = 1

        # Normalize the matrix
        row_scaling = np.reciprocal(row_norms)
        col_scaling = np.reciprocal(col_norms)
        D_row = sp.diags(row_scaling)  # Diagonal matrix for rows
        D_col = sp.diags(col_scaling)  # Diagonal matrix for columns

        A_normalized = D_row @ A_csr @ D_col  # Apply row and column normalization

        # Normalize objective coefficients and RHS
        obj_coeffs = obj_coeffs * col_norms
        rhs = rhs * row_scaling

        return A_normalized, obj_coeffs, rhs


    def normalize(self):
        """Normalize the matrix A, objective coefficients, and RHS."""
        #TODO: since floating point issues lead to problems, just exctract the ordering and not try to solve the normalized problem
        self.logger.debug("Starting normalization...")
        
        A_csr, self.obj_coeffs, self.rhs = self.normalize_matrix_consistently(
            self.A, self.obj_coeffs, self.rhs
        )
        self.A = A_csr

        self.logger.debug("Normalization completed.")
        self.logger.debug(f"Normalized matrix shape: {self.A.shape}, nnz: {self.A.nnz}")


    def generate_ordering(self):
        """Generate a consistent ordering of variables and constraints"""
        # Score and sort variables
        var_scores = np.abs(self.obj_coeffs)
        var_types = np.array([var.VType for var in self.vars])  # Extract variable types
        var_order = np.argsort(var_scores)

        # Reorder columns of A
        self.A = self.A[:, var_order]

        # Score and sort constraints
        constraint_scores = np.abs(self.A).sum(axis=1).A1
        constr_order = np.argsort(constraint_scores)

        # Reorder rows of A and RHS
        self.A = self.A[constr_order, :]
        self.rhs = self.rhs[constr_order]

        # Ensure variable types match the reordered variables
        self.vars = [self.vars[idx] for idx in var_order]
        self.obj_coeffs = self.obj_coeffs[var_order]
        var_types = var_types[var_order]

        return var_order, var_types, constr_order

    def get_canonical_form(self):
        """Generate canonical model"""
        self.normalize()
        var_order, var_types, constr_order = self.generate_ordering()

        # Create a new model
        canonical_model = gp.Model()
        new_vars = []

        for i, var_idx in enumerate(var_order):
            var = self.vars[var_idx]
            new_var = canonical_model.addVar(
                lb=self.vars[var_idx].LB,  # Ensure this aligns with the reordered index
                ub=self.vars[var_idx].UB,
                obj=self.obj_coeffs[i],
                vtype=var_types[i],  # Use the reordered variable types
                name=f"x{i+1}"
            )
            new_vars.append(new_var)

        A_csr = self.A.tocsr()

        for i, constr_idx in enumerate(constr_order):
            expr = gp.LinExpr()
            row = A_csr.getrow(i)

            for j, val in zip(row.indices, row.data):
                expr.add(new_vars[j], float(val))

            if self.constrs[constr_idx].Sense == GRB.LESS_EQUAL:
                canonical_model.addConstr(expr <= self.rhs[i])
            elif self.constrs[constr_idx].Sense == GRB.GREATER_EQUAL:
                canonical_model.addConstr(expr >= self.rhs[i])
            else:
                canonical_model.addConstr(expr == self.rhs[i])

        canonical_model.ModelSense = self.sense
        canonical_model.update()
        return canonical_model

    def _models_equivalent(self, model1, model2, tol=1e-6):
        """Check if two models are equivalent within tolerance with detailed logging"""
        self.logger.debug("Starting detailed model comparison...")
        
        # Check dimensions
        if model1.NumVars != model2.NumVars:
            self.logger.debug(f"Variable count mismatch: {model1.NumVars} vs {model2.NumVars}")
            return False
            
        if model1.NumConstrs != model2.NumConstrs:
            self.logger.debug(f"Constraint count mismatch: {model1.NumConstrs} vs {model2.NumConstrs}")
            return False
        
        # Compare matrices
        A1 = model1.getA().tocsr()
        A2 = model2.getA().tocsr()
        
        # Compare matrix properties
        if A1.nnz != A2.nnz:
            self.logger.debug(f"Non-zero elements mismatch: {A1.nnz} vs {A2.nnz}")
            return False

        # Compare matrix data
        if not np.allclose(A1.data, A2.data, atol=tol):
            diff_indices = ~np.isclose(A1.data, A2.data, atol=tol)
            diff_positions = np.where(diff_indices)[0]
            self.logger.debug("Matrix coefficient differences found:")
            for pos in diff_positions[:5]:  # Show first 5 differences
                self.logger.debug(f"Position {pos}: {A1.data[pos]} vs {A2.data[pos]}")
            return False
            
        # Compare objective coefficients
        obj1 = np.array([v.Obj for v in model1.getVars()])
        obj2 = np.array([v.Obj for v in model2.getVars()])
        
        if not np.allclose(obj1, obj2, atol=tol):
            diff_indices = ~np.isclose(obj1, obj2, atol=tol)
            var_indices = np.where(diff_indices)[0]
            self.logger.debug("Objective coefficient differences found:")
            for idx in var_indices[:5]:  # Show first 5 differences
                self.logger.debug(f"Variable {idx}: {obj1[idx]} vs {obj2[idx]}")
            return False

        # Compare variable bounds
        for i, (v1, v2) in enumerate(zip(model1.getVars(), model2.getVars())):
            if not (np.isclose(v1.LB, v2.LB, atol=tol) and 
                   np.isclose(v1.UB, v2.UB, atol=tol)):
                self.logger.debug(f"Bound mismatch for variable {i}:")
                self.logger.debug(f"LB: {v1.LB} vs {v2.LB}")
                self.logger.debug(f"UB: {v1.UB} vs {v2.UB}")
                return False
            
            if v1.VType != v2.VType:
                self.logger.debug(f"Variable type mismatch for variable {i}:")
                self.logger.debug(f"Type: {v1.VType} vs {v2.VType}")
                return False

        # Compare constraint senses and RHS
        for i, (c1, c2) in enumerate(zip(model1.getConstrs(), model2.getConstrs())):
            if c1.Sense != c2.Sense:
                self.logger.debug(f"Constraint sense mismatch for constraint {i}:")
                self.logger.debug(f"Sense: {c1.Sense} vs {c2.Sense}")
                return False
                
            if not np.isclose(c1.RHS, c2.RHS, atol=tol):
                self.logger.debug(f"RHS mismatch for constraint {i}:")
                self.logger.debug(f"RHS: {c1.RHS} vs {c2.RHS}")
                return False

        self.logger.debug("Models are equivalent within tolerance")
        return True

    def validate(self, canonical_from_orginal, canonical_from_permuted):
        """Validate canonical form invariance with detailed logging"""
        try:
            self.logger.debug("Starting model validation...")
            cf1 = canonical_from_orginal
            cf2 = canonical_from_permuted
            
            # Add comparison of matrix structures before detailed comparison
            A1 = cf1.getA()
            A2 = cf2.getA()
            self.logger.debug(f"Original canonical matrix: shape={A1.shape}, nnz={A1.nnz}")
            self.logger.debug(f"Permuted canonical matrix: shape={A2.shape}, nnz={A2.nnz}")
            
            result = self._models_equivalent(cf1, cf2)
            self.logger.debug(f"Validation result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False