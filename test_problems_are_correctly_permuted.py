from core.problem_permutator import ProblemPermutator
import gurobipy as gp
import numpy as np
from utils.gurobi_utils import get_Input_problem

# Test permutation integrity
input_problem = get_Input_problem()
original = gp.read(input_problem)
permutator = ProblemPermutator(input_problem)
permuted, _, _, _, _ = permutator.create_permuted_problem()


# Should print True for valid permutations
print(original.NumVars == permuted.NumVars)
print(original.NumConstrs == permuted.NumConstrs)
print(np.any([v.Obj != original.getVars()[i].Obj 
            for i, v in enumerate(permuted.getVars())]))