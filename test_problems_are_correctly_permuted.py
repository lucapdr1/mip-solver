from core.problem_permutator import ProblemPermutator
import gurobipy as gp
import numpy as np
from utils.config import INPUT_PROBLEM

# Test permutation integrity
original = gp.read(INPUT_PROBLEM)
permutator = ProblemPermutator(INPUT_PROBLEM)
permuted = permutator.create_permuted_problem()

# Should print True for valid permutations
print(original.NumVars == permuted.NumVars)
print(original.NumConstrs == permuted.NumConstrs)
print(np.any([v.Obj != original.getVars()[i].Obj 
            for i, v in enumerate(permuted.getVars())]))