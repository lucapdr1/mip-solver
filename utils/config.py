import os
import logging

# Read values from environment variables or use hardcoded defaults
LOG_LEVEL = os.getenv('LOG_LEVEL', logging.INFO)
INPUT_DIR = os.getenv('INPUT_DIR', "input/")

INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', INPUT_DIR + "dummy.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', INPUT_DIR + "example-min.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', INPUT_DIR + "30n20b8.mps")
#INPUT_PROBLEM = os.getenv('INPUT_PROBLEM', INPUT_DIR + "neos-4306827-ravan.mps")

NUMBER_OF_PERMUATATIONS = int(os.getenv('NUMBER_OF_PERMUATATIONS', 1))
NORMALIZATION_ACTIVE = os.getenv('NORMALIZATION_ACTIVE', 'False').lower() == 'true'

MATRICES_TO_CSV = os.getenv('MATRICES_TO_CSV', 'False').lower() == 'true'
LOG_MATRIX = os.getenv('LOG_MATRIX', 'False').lower() == 'true'
LOG_MODEL_COMPARISON = os.getenv('LOG_MODEL_COMPARISON', 'False').lower() == 'true'
PRODUCTION = os.getenv('PRODUCTION', 'True').lower() == 'true'

# Example of setting up logging
logging.basicConfig(level=LOG_LEVEL)

# Print or log the values to verify
print(f"LOG_LEVEL: {LOG_LEVEL}")
print(f"INPUT_DIR: {INPUT_DIR}")
print(f"INPUT_PROBLEM: {INPUT_PROBLEM}")
print(f"NUMBER_OF_PERMUATATIONS: {NUMBER_OF_PERMUATATIONS}")
print(f"NORMALIZATION_ACTIVE: {NORMALIZATION_ACTIVE}")
print(f"MATRICES_TO_CSV: {MATRICES_TO_CSV}")
print(f"LOG_MATRIX: {LOG_MATRIX}")
print(f"LOG_MODEL_COMPARISON: {LOG_MODEL_COMPARISON}")
print(f"PRODUCTION: {PRODUCTION}")