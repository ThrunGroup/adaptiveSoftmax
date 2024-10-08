# conv layer (CONST_* means it keeps the dimension)
CONST_KERNEL_SIZE = 5
CONST_STRIDE = 1
CONST_PADDING = 2
POOLING = 2    # halves the dimension    

NUM_CLASSES = 10
TRAINING_ITERATIONS = 50    # this is num epochs
EPOCH = 1000  # this is how many epochs actually used in training 
LERANING_RATE = 1e-2
BATCH_SIZE = 512  # NOTE: change this accordingly. Check free -h
PATIENCE = 10

# mnist 
MNIST = "mnist"
MNIST_IN_CHANNEL = 1  # grayscale
MNIST_OUT_CHANNEL = 256  # linear becomes 25088
MNIST_PATH = "mnl/data/mnist"

# eurosat
EUROSAT = "eurosat"
VGG19_IN_FEATURES = 25088
BLOCKS_NOT_FREEZING = 2  # set to zero if you only want to tune linear layer
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

# final experiments
GOOGLE_DRIVE_PREFIX = "https://drive.google.com/uc?id="
MNL_SCALING_POINTS = 5
MNIST_FINAL_PATH = "testing_mnist_out256_iter50_epochs14.npz"
EUROSAT_FINAL_PATH = "testing_eurosat_out25088_iter50_epochs1000.npz"
SEED = 42
NUM_QUERIES = 1000
FUDGE_TRAIN_SIZE = 200
DELTA = 0.01
EPS = 0.3
