import logging

LOG_LEVEL = logging.INFO
INPUT_DIR = "input/"
#INPUT_PROBLEM = INPUT_DIR + "dummy.mps" 
#INPUT_PROBLEM = INPUT_DIR + "30n20b8.mps" #secons solving time
#INPUT_PROBLEM = INPUT_DIR + "neos-4332801-seret.mps" #?solving time - too large
#INPUT_PROBLEM = INPUT_DIR + "neos-4306827-ravan.mps" # minutes solving time - visible decrese in performance varibility with canonical form
INPUT_PROBLEM = INPUT_DIR + "gen-ip054.mps" #production planning
NUMBER_OF_PERMUATATIONS = 5
NORMALIZATION_ACTIVE = False

MATRICES_TO_CSV = False
LOG_MATRIX = False
LOG_MODEL_COMPARISON =False