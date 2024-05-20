# conv layer (CONST_* means it keeps the dimension)
CONST_KERNEL_SIZE = 5
CONST_STRIDE = 1
CONST_PADDING = 2
POOLING = 2    # halves the dimension    

NUM_CLASSES = 10
TRAINING_ITERATIONS = 100    # this is num epochs
LERANING_RATE = 1e-3
BATCH_SIZE = 512  # NOTE: change this accordingly. Check free -h
PATIENCE = 5

# mnist 
MNIST = "mnist"
MNIST_IN_CHANNEL = 1  # grayscale
MNIST_OUT_CHANNEL = 256  # linear becomes 25088
MNIST_PATH = "mnl/data/mnist"

# eurosat
EUROSAT = "eurosat"
VGG19_IN_FEATURES = 25088
EUROSAT_PATH = "mnl/data/eurosat"

# A and x paths
MNL_WEIGHTS_DIR = 'mnl/weights'
MNL_XS_DIR = 'mnl/x_matrix'
MNL_ACC_DIR = "mnl/accuracies"
MNL_RESULTS_DIR = "experiments/mnl_results"

# test constants
NUM_EXPERIMENTS = 1
MNL_DELTA_SCALE = 3   # NOTE: this shouldn't change per test
MNL_TEST_BETA = 1.0
MNL_TEST_EPSILON = 0.1
MNL_TEST_DELTA = 0.01
MNL_TEST_TOPK = 1
MNL_TEST_SAMPLES_FOR_SIGMA = None   # this uses all d
MNL_TEST_IMPORTANCE = False
MNL_TEST_BUDGET_IMPROVEMENT = 1.0
MNL_TEST_SEED = 0

# mnist experiments
MNL_SCALING_POINTS = 5
MNIST_FINAL_PATH = "mnist_out256_iter10.npz"
EUROSAT_FINAL_PATH = ""

