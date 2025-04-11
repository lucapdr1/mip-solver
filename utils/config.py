import os
import logging

#LICENSE
WLSACCESSID = os.getenv('WLSACCESSID', '0f4125b4-a12f-4461-8ece-f8abc8da88c7')
WLSSECRET = os.getenv('WLSSECRET', '54d16d52-1e7f-44e9-a37f-f98498a2b836')
LICENSEID = os.getenv('LICENSEID', '2648812')

# Read values from environment variables or use hardcoded defaults
LOG_LEVEL = os.getenv('LOG_LEVEL', logging.INFO)
INPUT_DIR = os.getenv('INPUT_DIR', "input/")
OUTPUT_DIR = os.getenv('OUTPUT_DIR', "experiments/")

#INPUT_PROBLEM =  os.getenv('INPUT_PROBLEM', "dummy.mps")
#INPUT_PROBLEM =  os.getenv('INPUT_PROBLEM', "dummy_constrT.mps")
#INPUT_PROBLEM =  os.getenv('INPUT_PROBLEM', "dummy_with_bounds.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "example-min.mps")
INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "neos-911970.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "exp-1-500-5-5.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "gen-ip054.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "markshare2.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "30n20b8.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "neos-4306827-ravan.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "net12.mps")

PERMUTE_ORIGINAL = os.getenv('PERMUTE_ORIGINAL', 'True').lower() == 'true'
PERMUTE_SEED = int(os.getenv('PERMUTE_SEED', 12345))
PERMUTE_GRANULARITY_K = os.getenv('PERMUTE_GRANULARITY_K', "all")# "all" to permute all, any integer K e.g 10 to define 1O subblocks
APPLY_DEC = os.getenv('APPLY_DEC', 'True').lower() == 'true'
NUMBER_OF_PERMUTATIONS = int(os.getenv('NUMBER_OF_PERMUTATIONS', 3))
NORMALIZATION_ACTIVE = os.getenv('NORMALIZATION_ACTIVE', 'False').lower() == 'true'
SCALING_ACTIVE = os.getenv('SCALING_ACTIVE', 'False').lower() == 'true' 
CUSTOM_RULES_ACTIVE = os.getenv('CUSTOM_RULES_ACTIVE', 'False').lower() == 'true'
BLOCK_ORDERING_ACTIVE = os.getenv('BLOCK_ORDERING_ACTIVE', 'False').lower() == 'true'

MATRICES_TO_CSV = os.getenv('MATRICES_TO_CSV', 'False').lower() == 'true'
LOG_MATRIX = os.getenv('LOG_MATRIX', 'False').lower() == 'true'
LOG_MODEL_COMPARISON = os.getenv('LOG_MODEL_COMPARISON', 'False').lower() == 'true'
PRODUCTION = os.getenv('PRODUCTION', 'False').lower() == 'true'
RECURSIVE_RULES = os.getenv('RECURSIVE_RULES', 'True').lower() == 'true'
DISABLE_SOLVING = os.getenv('DISABLE_SOLVING', 'False').lower() == 'true'
MAX_SOLVE_TIME = int(os.getenv('MAX_SOLVE_TIME', 3600))
NUMBER_OF_THREADS = int(os.getenv('NUMBER_OF_THREADS', 4))

#AWS
BUCKET_NAME = os.getenv('BUCKET_NAME', 'lucapolimi-experiments')

# Example of setting up logging
logging.basicConfig(level=LOG_LEVEL)