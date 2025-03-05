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
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "example-min.mps")
INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "gen-ip054.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "markshare2.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "30n20b8.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "neos-4306827-ravan.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', "net12.mps")

NUMBER_OF_PERMUTATIONS = int(os.getenv('NUMBER_OF_PERMUTATIONS', 1))
NORMALIZATION_ACTIVE = os.getenv('NORMALIZATION_ACTIVE', 'False').lower() == 'true'
SCALING_ACTIVE = os.getenv('SCALING_ACTIVE', 'False').lower() == 'true' 

MATRICES_TO_CSV = os.getenv('MATRICES_TO_CSV', 'False').lower() == 'true'
LOG_MATRIX = os.getenv('LOG_MATRIX', 'False').lower() == 'true'
LOG_MODEL_COMPARISON = os.getenv('LOG_MODEL_COMPARISON', 'False').lower() == 'true'
PRODUCTION = os.getenv('PRODUCTION', 'False').lower() == 'true'
RECURSIVE_RULES = os.getenv('RECURSIVE_RULES', 'True').lower() == 'true'
DISABLE_SOLVING = os.getenv('DISABLE_SOLVING', 'False').lower() == 'true'

#AWS
BUCKET_NAME = os.getenv('BUCKET_NAME', 'lucapolimi-experiments')

# Example of setting up logging
logging.basicConfig(level=LOG_LEVEL)