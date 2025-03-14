import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
from utils.logging_handler import LoggingHandler
from utils.config import NORMALIZATION_ACTIVE


class CanonicalFormGenerator:
    def __init__(self, gp_env, model, ordering_rule):
        self.original_model = model
        self.model = model.copy()
        self.ordering_rule = ordering_rule
        self.logger = LoggingHandler().get_logger()
        self._initialize_structures()
        self.gp_env = gp_env
        
    def _initialize_structures(self):
        """Extract model components with integrity checks"""
        self.vars = np.array(self.model.getVars(), dtype=object)
        self.constrs = np.array(self.model.getConstrs(), dtype=object)
        self.A = self.model.getA()  # Constraint matrix in sparse format
        self.A_csr = self.A.tocsr()
        self.A_csc = self.A.tocsc()

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
        self.original_bounds = np.array([(var.LB, var.UB) for var in self.vars], dtype=float)
    
    def generate_ordering(self):
        """Generate a consistent ordering of variables and constraints."""
        # Score and sort variables
        var_scores = self.ordering_rule.score_variables(
            self.vars, self.obj_coeffs, self.original_bounds, self.A, self.A_csc, self.A_csr, self.constrs, self.rhs
        )
        # Log variable scores in one shot if debug is enabled.
        self.logger.lazy_debug("Variable scores before ordering: %s", var_scores)
        
        # Get variable types once.
        var_types = np.array([var.VType for var in self.vars])
        
        # Compute ordering using NumPy's argsort.
        var_order = np.argsort(var_scores)
        self.logger.lazy_debug("Variable ordering: %s", var_order)
        
        # Log original variable bounds (combine into one string to avoid per-item loop).
        bounds_str = ", ".join(
            f"Var {i} (Type: {'Continuous' if var.VType == GRB.CONTINUOUS else 'Integer' if var.VType == GRB.INTEGER else 'Binary'}): [{var.LB}, {var.UB}]"
            for i, var in enumerate(self.vars)
        )
        self.logger.lazy_debug("Original bounds before reordering: %s", bounds_str)
        
        # Reorder columns of A
        self.A = self.A_csc[:, var_order]

        # Score and sort constraints.
        constraint_scores = self.ordering_rule.score_constraints(
            self.vars, self.obj_coeffs, self.original_bounds, self.A, self.A_csc, self.A_csr, self.constrs, self.rhs
        )
        self.logger.lazy_debug("Constraint scores before ordering: %s", constraint_scores)
        
        constr_order = np.argsort(constraint_scores)
        self.logger.lazy_debug("Constraint ordering: %s", constr_order)

        # Log original constraints (build one combined string).
        constr_str = ", ".join(
            f"Constr {i}: { {'<': '<=', '>': '>=', '=': '='}[constr.Sense] } {self.rhs[i]}"
            for i, constr in enumerate(self.constrs)
        )
        self.logger.lazy_debug("Original constraints before reordering: %s", constr_str)
        
        # Reorder rows of A and RHS.
        self.A = self.A_csr[constr_order, :]
        self.rhs = self.rhs[constr_order]

        # Reorder variables, objective coefficients, and bounds.
        self.vars = [self.vars[idx] for idx in var_order]
        self.obj_coeffs = self.obj_coeffs[var_order]
        self.original_bounds = [self.original_bounds[idx] for idx in var_order]
        var_types = var_types[var_order]

        return var_order, var_types, constr_order


    def get_canonical_form(self):
        """Generate canonical model"""
        var_order, var_types, constr_order = self.generate_ordering()

        # Create a new model
        canonical_model = gp.Model(env=self.gp_env)
        new_vars = []

        for i, var_idx in enumerate(var_order):
            var = self.vars[var_idx]
            ord_lb, ord_ub = self.original_bounds[i]
            new_var = canonical_model.addVar(
                lb=ord_lb,  # Ensure this aligns with the reordered index
                ub=ord_ub,
                obj=self.obj_coeffs[i],
                vtype=var_types[i],  # Use the reordered variable types
                name=f"x{i+1}"
            )
            new_vars.append(new_var)

        indptr = self.A_csr.indptr
        indices = self.A_csr.indices
        data = self.A_csr.data

        for i in range(self.A_csr.shape[0]):
            expr = gp.LinExpr()
            for idx in range(indptr[i], indptr[i+1]):
                j = indices[idx]
                expr.add(new_vars[j], float(data[idx]))
            # Use constr_order[i] to index into self.constrs for the sense and RHS
            constr = self.constrs[constr_order[i]]
            if constr.Sense == GRB.LESS_EQUAL:
                canonical_model.addConstr(expr <= self.rhs[i])
            elif constr.Sense == GRB.GREATER_EQUAL:
                canonical_model.addConstr(expr >= self.rhs[i])
            else:
                canonical_model.addConstr(expr == self.rhs[i])

        canonical_model.ModelSense = self.sense
        canonical_model.update()
        return canonical_model, var_order, constr_order

    def _models_equivalent(self, model1, model2, tol=1e-6):
        """Check if two models are equivalent within tolerance with detailed logging"""
        self.logger.lazy_debug("Starting detailed model comparison...")
        
        # Check dimensions
        if model1.NumVars != model2.NumVars:
            self.logger.lazy_debug(f"Variable count mismatch: {model1.NumVars} vs {model2.NumVars}")
            return False
            
        if model1.NumConstrs != model2.NumConstrs:
            self.logger.lazy_debug(f"Constraint count mismatch: {model1.NumConstrs} vs {model2.NumConstrs}")
            return False
        
        # Compare matrices
        A1 = model1.getA().tocsr()
        A2 = model2.getA().tocsr()
        
        # Compare matrix properties
        if A1.nnz != A2.nnz:
            self.logger.lazy_debug(f"Non-zero elements mismatch: {A1.nnz} vs {A2.nnz}")
            return False

        # Compare matrix data
        if not np.allclose(A1.data, A2.data, atol=tol):
            diff_indices = ~np.isclose(A1.data, A2.data, atol=tol)
            diff_positions = np.where(diff_indices)[0]
            self.logger.lazy_debug("Matrix coefficient differences found:")
            for pos in diff_positions[:5]:  # Show first 5 differences
                self.logger.lazy_debug(f"Position {pos}: {A1.data[pos]} vs {A2.data[pos]}")
            return False
            
        # Compare objective coefficients
        obj1 = np.array([v.Obj for v in model1.getVars()])
        obj2 = np.array([v.Obj for v in model2.getVars()])
        
        if not np.allclose(obj1, obj2, atol=tol):
            diff_indices = ~np.isclose(obj1, obj2, atol=tol)
            var_indices = np.where(diff_indices)[0]
            self.logger.lazy_debug("Objective coefficient differences found:")
            for idx in var_indices[:5]:  # Show first 5 differences
                self.logger.lazy_debug(f"Variable {idx}: {obj1[idx]} vs {obj2[idx]}")
            return False

        # Compare variable bounds
        for i, (v1, v2) in enumerate(zip(model1.getVars(), model2.getVars())):
            if not (np.isclose(v1.LB, v2.LB, atol=tol) and 
                   np.isclose(v1.UB, v2.UB, atol=tol)):
                self.logger.lazy_debug(f"Bound mismatch for variable {i}:")
                self.logger.lazy_debug(f"LB: {v1.LB} vs {v2.LB}")
                self.logger.lazy_debug(f"UB: {v1.UB} vs {v2.UB}")
                return False
            
            if v1.VType != v2.VType:
                self.logger.lazy_debug(f"Variable type mismatch for variable {i}:")
                self.logger.lazy_debug(f"Type: {v1.VType} vs {v2.VType}")
                return False

        # Compare constraint senses and RHS
        for i, (c1, c2) in enumerate(zip(model1.getConstrs(), model2.getConstrs())):
            if c1.Sense != c2.Sense:
                self.logger.lazy_debug(f"Constraint sense mismatch for constraint {i}:")
                self.logger.lazy_debug(f"Sense: {c1.Sense} vs {c2.Sense}")
                return False
                
            if not np.isclose(c1.RHS, c2.RHS, atol=tol):
                self.logger.lazy_debug(f"RHS mismatch for constraint {i}:")
                self.logger.lazy_debug(f"RHS: {c1.RHS} vs {c2.RHS}")
                return False

        self.logger.lazy_debug("Models are equivalent within tolerance")
        return True

    def validate(self, canonical_from_orginal, canonical_from_permuted):
        """Validate canonical form invariance with detailed logging"""
        try:
            self.logger.lazy_debug("Starting model validation...")
            cf1 = canonical_from_orginal
            cf2 = canonical_from_permuted
            
            # Add comparison of matrix structures before detailed comparison
            A1 = cf1.getA()
            A2 = cf2.getA()
            self.logger.lazy_debug(f"Original canonical matrix: shape={A1.shape}, nnz={A1.nnz}")
            self.logger.lazy_debug(f"Permuted canonical matrix: shape={A2.shape}, nnz={A2.nnz}")
            
            result = self._models_equivalent(cf1, cf2)
            self.logger.lazy_debug(f"Validation result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False