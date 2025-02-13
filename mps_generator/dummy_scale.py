import gurobipy as gp
from gurobipy import GRB

from utils.gurobi_utils import init_gurobi_env

#env
gp_env = init_gurobi_env()
# Create a new Gurobi model
model = gp.Model("dummy_scale", env=gp_env)

# Define variables
x1 = model.addVar(vtype=GRB.CONTINUOUS, name="x1")
x3 = model.addVar(vtype=GRB.CONTINUOUS, name="x3")

# Set the objective function
model.setObjective(2*x1 + 6*x3, GRB.MINIMIZE)

# Add constraints
model.addConstr(4*x1<= 10, name="C1")
model.addConstr(2*x3 <= 20, name="C2")

# Update the model
model.update()

# Write the model to an MPS file
mps_filename = "dummy_scale.mps"
model.write(mps_filename)

print(f"MPS file '{mps_filename}' has been created.")
