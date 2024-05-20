# algorithm constants
TOP_K = 1
BETA = 1.0
DEFAULT_EPSILON = 0.1   
DEFAULT_DELTA = 0.01
EPSILON_SCALE = 4   # refer to algo 1 in paper
DELTA_SCALE = 3
BATCH_SIZE = 128
IMPORTANCE = True  # toggle this for importance sampling
SIGMA_BUFFER = 1e-3
DEFAULT_CI_INIT = 1/4
DEFAULT_CI_DECAY = 1/2
DEFAULT_VAR_PULL_INIT = 16
DEFAULT_VAR_PULL_INCR = 3/2
TUNE_EXP_FUDGE_LOW = -6
TUNE_EXP_FUDGE_HIGH = 0

# budget constants
UNI_CONST = 3e-3
F_ORDER_CONST = 1
S_ORDER_CONST = 1
VERBOSE = True  # set flag for more stats

# debugging constants (will write to log)
DEBUG = False
DEV_BY = 1  # std deviations
DEV_RATIO = 0.5  # setting smaller leads to more outliers
NUM_BINS = 10


# pytest constants
NUM_TESTS = 100
NUM_ROWS = 10
NUM_COLS = int(3e4)
BUDGET_IMPROVEMENT = 1.0    # improve the budget by

TEST_BETA = 1.0
TEST_EPSILON = 0.1
TEST_DELTA = 0.01
TEST_TOPK = 1
TEST_SEED = 0
TEST_SAMPLES_FOR_SIGMA = None   # this uses all d
TEST_IMPORTANCE = False

# elements of random mu (TODO: this assumes normalized?)
TEST_MU_LOWER = 0
TEST_MU_UPPER = 10
SCALING_POINTS = 10
NUM_TRIALS = 10

