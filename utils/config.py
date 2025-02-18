import os
import logging

#LICENSE
WLSACCESSID = os.getenv('WLSACCESSID')
WLSSECRET = os.getenv('WLSSECRET')
LICENSEID = os.getenv('LICENSEID')

# Read values from environment variables or use hardcoded defaults
LOG_LEVEL = os.getenv('LOG_LEVEL', logging.INFO)
INPUT_DIR = os.getenv('INPUT_DIR', "input/")
OUTPUT_DIR = os.getenv('OUTPUT_DIR', "experiments/")

#INPUT_PROBLEM =  os.getenv('INPUT_PROBLEM', "dummy.mps")
#INPUT_PROBLEM =  os.getenv('INPUT_PROBLEM', "dummy_constrT.mps")
#INPUT_PROBLEM =  os.getenv('INPUT_PROBLEM', "dummy_with_bounds.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "example-min.mps") gen-ip054.mps
INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "gen-ip054.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "30n20b8.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "neos-4306827-ravan.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "net12.mps")

NUMBER_OF_PERMUTATIONS = int(os.getenv('NUMBER_OF_PERMUTATIONS', 3))
NORMALIZATION_ACTIVE = os.getenv('NORMALIZATION_ACTIVE', 'False').lower() == 'true'
SCALING_ACTIVE = os.getenv('SCALING_ACTIVE', 'False').lower() == 'true' 

MATRICES_TO_CSV = os.getenv('MATRICES_TO_CSV', 'False').lower() == 'true'
LOG_MATRIX = os.getenv('LOG_MATRIX', 'False').lower() == 'true'
LOG_MODEL_COMPARISON = os.getenv('LOG_MODEL_COMPARISON', 'False').lower() == 'true'
PRODUCTION = os.getenv('PRODUCTION', 'False').lower() == 'true'
RECURSIVE_RULES = os.getenv('RECURSIVE_RULES', 'True').lower() == 'true'
DISABLE_SOLVING = os.getenv('DISABLE_SOLVING', 'True').lower() == 'true'

#AWS
BUCKET_NAME = os.getenv('BUCKET_NAME', 'lucapolimi-experiments')

# Example of setting up logging
logging.basicConfig(level=LOG_LEVEL)

# Print or log the values to verify
print(f"LOG_LEVEL: {LOG_LEVEL}")
print(f"INPUT_DIR: {INPUT_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"INPUT_PROBLEM: {INPUT_PROBLEM}")
print(f"NUMBER_OF_PERMUTATIONS: {NUMBER_OF_PERMUTATIONS}")
print(f"NORMALIZATION_ACTIVE: {NORMALIZATION_ACTIVE}")
print(f"MATRICES_TO_CSV: {MATRICES_TO_CSV}")
print(f"LOG_MATRIX: {LOG_MATRIX}")
print(f"LOG_MODEL_COMPARISON: {LOG_MODEL_COMPARISON}")
print(f"PRODUCTION: {PRODUCTION}")
print(f"DISABLE_SOLVING: {DISABLE_SOLVING}")