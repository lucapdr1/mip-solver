from core.problem_transform.problem_permutator import ProblemPermutator
import gurobipy as gp
import numpy as np
from utils.gurobi_utils import get_Input_problem, init_gurobi_env

# Test permutation integrity
gp_env = init_gurobi_env()
input_problem = get_Input_problem()
original = gp.read(input_problem, env=gp_env)
permutator = ProblemPermutator(gp_env, original)
permuted, _, _, _, _ = permutator.create_permuted_problem(1,12345)


# Should print True for valid permutations
print(original.NumVars == permuted.NumVars)
print(original.NumConstrs == permuted.NumConstrs)
