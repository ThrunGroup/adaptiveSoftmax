BATCH_SIZE = 128
TOP_K = 1
BETA = 1.0   # this is roughly inverse of sigma 
DEFAULT_EPSILON = 0.1
DEFAULT_DELTA = 0.01
PROFILE = False
RETURN_STAGE_BUDGETS = False  # TODO (@wonjun): this constant unused?
OPTIMIZE_CONSTANTS = False
VERBOSE = True     # this just prints out the budgets. set DEBUG flag for more stats

# debugging constants (will write to log)
DEBUG = True
DEV_BY = 1  # std deviations  
DEV_RATIO = 0.5 # setting smaller leads to more outliers
NUM_BINS = 10

# constants for budget  
UNI_CONST = 1
F_ORDER_CONST = 1
S_ORDER_CONST = 1
