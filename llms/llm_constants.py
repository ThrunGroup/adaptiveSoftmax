# dataset
WIKITEXT_DATASET = "wikitext"
WIKITEXT_BETA = 1.0

# paths
LLM_WEIGHTS_DIR = "llms/weights"
LLM_XS_DIR = "llms/x_matrix"
LLM_TARGET_IDS_PATH = "llms/target_ids"

# gpt constants
GPT2 = "gpt2"
GPT_EPS = 0.1
GPT_DELTA_ERROR = 0.1
GPT_FINAL_HIDDEN_LAYER_NAME = "transformer"

# llama constants
LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B"
LLAMA_FINAL_HIDDEN_LAYER_NAME = "norm"  # llama3 applies layer norm 


# other
CONTEXT_WINDOW_STRIDE = 512   

# testing constants
LLM_TEST_EPSILON = 0.1
LLM_TEST_DELTA = 0.1 
LLM_TEST_BETA = 1.0
LLM_TEST_IMPORTANCE = True
LLM_TEST_TOPK = 1
LLM_TEST_BUDGET_IMPROVEMENT = 1.0
LLM_DELTA_SCALE = 3

NUM_EXPERIMENTS = 10