# test_problem_scaler_negative.py

from core.problem_scaler import ProblemScaler
import gurobipy as gp
import numpy as np
from utils.gurobi_utils import get_Input_problem, init_gurobi_env

# Initialize Gurobi environment and read the input problem.
gp_env = init_gurobi_env()
input_problem = get_Input_problem()
original = gp.read(input_problem, env=gp_env)

# Create a ProblemScaler instance.
scaler = ProblemScaler(gp_env, original)

# Determine the number of variables and constraints.
num_vars = original.NumVars
num_constrs = original.NumConstrs

# --- Create explicit scaling vectors with negative scaling ---
# For testing, we assign a negative scaling factor for the first constraint and first variable.
row_scales = np.ones(num_constrs)
col_scales = np.ones(num_vars)
if num_constrs > 0:
    row_scales[0] = -2.0   # Negative scaling: should flip constraint sense.
if num_vars > 0:
    col_scales[0] = -3.0   # Negative scaling: variable bounds swap and objective is divided by -3.

# Create a scaled model using the explicit scaling factors.
scaled, used_row_scales, used_col_scales, D_row, D_col = scaler.create_scaled_problem_with_scales(row_scales, col_scales)

# --- Test variable scaling (including negative scaling) ---
if num_vars > 0:
    orig_var = original.getVars()[0]
    scaled_var = scaled.getVars()[0]
    
    # Check objective coefficient: expected new_obj = original_obj / (col_scale)
    expected_obj = orig_var.Obj / used_col_scales[0]
    print("First variable negative scaling test:")
    print("  Original obj:   ", orig_var.Obj)
    print("  Scaled obj:     ", scaled_var.Obj)
    print("  Expected obj:   ", expected_obj)
    print("  Objective correct:", np.isclose(scaled_var.Obj, expected_obj))
    
    # Check bounds: new_lb = min(s * LB, s * UB), new_ub = max(s * LB, s * UB)
    expected_lb = min(used_col_scales[0] * orig_var.LB, used_col_scales[0] * orig_var.UB)
    expected_ub = max(used_col_scales[0] * orig_var.LB, used_col_scales[0] * orig_var.UB)
    print("  Original bounds:", orig_var.LB, orig_var.UB)
    print("  Scaled bounds:  ", scaled_var.LB, scaled_var.UB)
    print("  Expected bounds:", expected_lb, expected_ub)
    bounds_correct = np.isclose(scaled_var.LB, expected_lb) and np.isclose(scaled_var.UB, expected_ub)
    print("  Bounds correct:", bounds_correct)
else:
    print("No variables in the model to test.")

# --- Test constraint scaling (including negative scaling) ---
if num_constrs > 0:
    orig_constr = original.getConstrs()[0]
    scaled_constr = scaled.getConstrs()[0]
    
    # Check right-hand side: expected new_rhs = row_scale * original_RHS.
    expected_rhs = used_row_scales[0] * orig_constr.RHS
    print("\nFirst constraint negative scaling test:")
    print("  Original RHS:   ", orig_constr.RHS)
    print("  Scaled RHS:     ", scaled_constr.RHS)
    print("  Expected RHS:   ", expected_rhs)
    print("  RHS correct:    ", np.isclose(scaled_constr.RHS, expected_rhs))
    
    # Check constraint sense: if row scaling is negative, the sense should flip.
    if orig_constr.Sense == gp.GRB.LESS_EQUAL:
        expected_sense = gp.GRB.GREATER_EQUAL
    elif orig_constr.Sense == gp.GRB.GREATER_EQUAL:
        expected_sense = gp.GRB.LESS_EQUAL
    else:  # Equality constraints remain unchanged.
        expected_sense = gp.GRB.EQUAL
    print("  Original sense: ", orig_constr.Sense)
    print("  Scaled sense:   ", scaled_constr.Sense)
    print("  Expected sense: ", expected_sense)
    print("  Sense correct:  ", scaled_constr.Sense == expected_sense)
else:
    print("No constraints in the model to test.")

# --- Overall model dimensions test ---
print("\nOverall model dimensions test:")
print("  Same number of variables:", original.NumVars == scaled.NumVars)
print("  Same number of constraints:", original.NumConstrs == scaled.NumConstrs)
