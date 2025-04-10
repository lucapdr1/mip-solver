import sys
from utils.dec_preprocess import DecGenerator
from utils.gurobi_utils import init_gurobi_env, get_Input_problem
from utils.config import NUMBER_OF_PERMUTATIONS

if __name__ == "__main__":
    try:
        gp_env = init_gurobi_env()
        input_problem = get_Input_problem()

        generator = DecGenerator(gp_env,input_problem)
        results = generator.create_decmpositions(NUMBER_OF_PERMUTATIONS)
    except Exception as e:
        print(f"Preprocess failed: {e}")
