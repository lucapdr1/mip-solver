import gurobipy as gp
import os
from utils.config import WLSACCESSID, WLSSECRET, LICENSEID, INPUT_DIR, INPUT_PROBLEM

# Initialize the Gurobi environment
def init_gurobi_env():
    # Create a dictionary of parameters
    params = {
        'WLSACCESSID': WLSACCESSID,
        'WLSSECRET':WLSSECRET,
        'LICENSEID': int(LICENSEID) if LICENSEID else None
    }

    # Remove None values from parameters
    params = {k: v for k, v in params.items() if v is not None}

    # Initialize the Gurobi environment with parameters
    env = gp.Env(params=params)
    env.start()
    return env

def get_Input_problem():
    return INPUT_DIR + INPUT_PROBLEM
