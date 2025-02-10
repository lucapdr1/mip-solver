import gurobipy as gp
from gurobipy import GRB

from utils.gurobi_utils import init_gurobi_env

#env
gp_env = init_gurobi_env()
# Create a new Gurobi model
model = gp.Model("dummy", env=gp_env)

# Define variables
x1 = model.addVar(vtype=GRB.INTEGER, name="x1")
x2 = model.addVar(vtype=GRB.INTEGER, name="x2")
x3 = model.addVar(vtype=GRB.CONTINUOUS, name="x3")
x4 = model.addVar(vtype=GRB.CONTINUOUS, name="x4")

# Set the objective function
model.setObjective(4*x1 + 5*x2 + 2*x3 + 3*x4, GRB.MINIMIZE)

# Add constraints
model.addConstr(2*x1 + 3*x3 <= 6, name="C1")
model.addConstr(x2 - x4 <= 0, name="C2")
model.addConstr(x1 + x2 + x3 + x4 <= 10, name="C3")

# Update the model
model.update()

# Write the model to an MPS file
mps_filename = "dummy.mps"
model.write(mps_filename)

print(f"MPS file '{mps_filename}' has been created.")
