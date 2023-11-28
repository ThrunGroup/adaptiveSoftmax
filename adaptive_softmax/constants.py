BATCH_SIZE = 128
TOP_K = 1
BETA = 1.0   # this is roughly inverse of sigma 
DEFAULT_EPSILON = 0.1
DEFAULT_DELTA = 0.01
PROFILE = False
RETURN_STAGE_BUDGETS = False  # TODO (@wonjun): this constant unused?
OPTIMIZE_CONSTANTS = False

PRECOMPUTE = False
DEV_RATIO = 0.3   # setting this smaller means more outliers

# TODO: decompose this into more constants
# debugging should give information on the following:
#   1. decomposition of budget for different stages 
#   2. decomposition of error probability (i.e. where is algorithm failing?)
#   3. how "heavy" are heavy hitters -> sigma before/after 
PLOT_VARIANCE = False
PLOT_BUDGET = False

# constants for budget  

"""
UNI_CONST = 8e-4 # T0 = d/10
F_ORDER_CONST = 1e2 
S_ORDER_CONST = 1e10
"""


UNI_CONST = 1
F_ORDER_CONST = 1
S_ORDER_CONST = 1
