from core.problem_permutator import ProblemPermutator
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import hashlib
import scipy.sparse as sp
from networkx import Graph, community

# Test permutation integrity
original = gp.read("input/dummy.mps")
permutator = ProblemPermutator("input/dummy.mps")
permuted = permutator.create_permuted_problem()

# Should print True for valid permutations
print(original.NumVars == permuted.NumVars)
print(original.NumConstrs == permuted.NumConstrs)
print(np.any([v.Obj != original.getVars()[i].Obj 
            for i, v in enumerate(permuted.getVars())]))