# dataset
WIKITEXT_DATASET = "wikitext"
WIKITEXT_BETA = 1.0
PENN_TREEBANK_DATASET = "penn_treebank"

# paths
LLM_WEIGHTS_DIR = "llms/weights"
LLM_XS_DIR = "llms/x_matrix"
LLM_RESULTS_DIR = "experiments/llm_results"

# llm constants
GPT2 = "gpt2"
LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B"
MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
GEMMA_7B = "google/gemma-7b"

# other
CONTEXT_WINDOW_STRIDE = 256
MAX_LENGTH = 1024  # cap this due to OOM errors

# testing constants
LLM_TEST_EPSILON = 0.1
LLM_TEST_DELTA = 0.1 
LLM_TEST_BETA = 1.0
LLM_TEST_IMPORTANCE = True
LLM_TEST_TOPK = 1
LLM_TEST_BUDGET_IMPROVEMENT = 1.0
LLM_DELTA_SCALE = 3

# final experimental results
SEED = 42
DELTA = 0.01
EPS = 0.3
TRAIN_SIZE = 100
NUM_EXPERIMENTS = 10
NUM_QUERY = 1000