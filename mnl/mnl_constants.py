# conv layer (CONST_* means it keeps the dimension)
CONST_KERNEL_SIZE = 3
CONST_STRIDE = 1
CONST_PADDING = 1
POOLING = 2    # halves the dimension    
DROPOUT = 0.25

NUM_CLASSES = 10
TRAINING_ITERATIONS = 1
BATCH_SIZE = 256

# mnist 
MNIST = "mnist"
MNIST_IN_CHANNEL = 1  # grayscale
MNIST_OUT_CHANNEL = 32
MNIST_PATH = "mnl/data/mnist"

# eurosat
EUROSAT = "eurosat"
EUROSAT_IN_CHANNEL = 3  # RGB
EUROSAT_OUT_CHANNEL = 32
EUROSAT_PATH = "mnl/data/eurosat"
EUROSAT_DATAPOINTS = 27000

# A and x paths
MNL_WEIGHTS_DIR = 'mnl/weights'
MNL_XS_DIR = 'mnl/x_matrix'

# test constants
NUM_EXPERIMENTS = 1
MNL_DELTA_SCALE = 3   
MNL_TEST_BETA = 1.0
MNL_TEST_EPSILON = 0.1
MNL_TEST_DELTA = 0.01
MNL_TEST_TOPK = 1
MNL_TEST_SAMPLES_FOR_SIGMA = None   # this uses all d
MNL_TEST_IMPORTANCE = False
MNL_TEST_BUDGET_IMPROVEMENT = 1.0
MNL_TEST_SEED = 0

# mnist tests
MNL_TEST_MNIST_LINEAR = 3136 # out_channel = 16

