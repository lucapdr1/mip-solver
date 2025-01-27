import gurobipy as gp
from gurobipy import GRB

# Create a new Gurobi model
model = gp.Model("dummy")

# Define variables with some bounds
x1 = model.addVar(vtype=GRB.INTEGER, lb=0, ub=5, name="x1")  # integer, bounded between 0 and 5
x2 = model.addVar(vtype=GRB.INTEGER, lb=1, ub=10, name="x2")  # integer, bounded between 1 and 10
x3 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=7.5, name="x3")  # continuous, bounded between 0 and 7.5
x4 = model.addVar(vtype=GRB.CONTINUOUS, lb=-2, ub=2, name="x4")  # continuous, bounded between -2 and 2

# Set the objective function
model.setObjective(4*x1 + 5*x2 + 2*x3 + 3*x4, GRB.MINIMIZE)

# Add constraints
model.addConstr(2*x1 + 3*x3 <= 6, name="C1")
model.addConstr(x2 - x4 <= 0, name="C2")
model.addConstr(x1 + x2 + x3 + x4 <= 10, name="C3")

# Update the model
model.update()

# Write the model to an MPS file
mps_filename = "dummy_with_bounds.mps"
model.write(mps_filename)

print(f"MPS file '{mps_filename}' has been created.")