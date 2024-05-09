import pytest
from .test_utils import epsilon_check, delta_check
from llms.llm_utils import load_llm_matrices
from llms.llm_constants import (
    GPT2,
    LLAMA_3_8B,
    MISTRAL_7B,
    GEMMA_7B,
    CONTEXT_WINDOW_STRIDE,
    LLM_TEST_EPSILON,
    LLM_TEST_DELTA,
    LLM_TEST_BETA,
    LLM_TEST_IMPORTANCE,
    LLM_TEST_TOPK,
    LLM_TEST_BUDGET_IMPROVEMENT,
    LLM_DELTA_SCALE,
    NUM_EXPERIMENTS,
    WIKITEXT_DATASET,
    PENN_TREEBANK_DATASET,
)

llm_constants = {
    'stride': CONTEXT_WINDOW_STRIDE,
    'eps': LLM_TEST_EPSILON,
    'delta': LLM_TEST_DELTA,
    'temp': LLM_TEST_BETA,
    'query_importance': LLM_TEST_IMPORTANCE,
    'top_k': LLM_TEST_TOPK,
    'num_experiments': NUM_EXPERIMENTS
}

# Parametrize the fixture to set up llm_constants with different models
@pytest.fixture(params=[GPT2, LLAMA_3_8B, MISTRAL_7B, GEMMA_7B], ids=['GPT2', 'LLAMA_3_8B', 'MISTRAL_7B', 'GEMMA_7B'])
def setup_llm_constants(request):
    llm_constants['model'] = request.param
    return llm_constants

# Parametrize tests to run for each combination of dataset
#@pytest.mark.parametrize("dataset", [WIKITEXT_DATASET, PENN_TREEBANK_DATASET], ids=['WIKITEXT', 'PENN'])
@pytest.mark.parametrize("dataset", [WIKITEXT_DATASET], ids=['WIKITEXT'])
def test_eps(dataset, setup_llm_constants):
    in_bounds, budget, naive_budget = epsilon_check(
        dataset, 
        load_llm_matrices, 
        **setup_llm_constants
    )
    assert in_bounds
    assert budget < naive_budget / LLM_TEST_BUDGET_IMPROVEMENT

@pytest.mark.parametrize("dataset", [WIKITEXT_DATASET, PENN_TREEBANK_DATASET], ids=['WIKITEXT', 'PENN'])
def test_delta(dataset, setup_llm_constants):
    total_wrong, total_budget, naive_budget = delta_check(
        dataset, 
        load_llm_matrices, 
        **setup_llm_constants
    )
    assert total_wrong / NUM_EXPERIMENTS < LLM_TEST_DELTA / LLM_DELTA_SCALE
    assert total_budget < naive_budget / LLM_TEST_BUDGET_IMPROVEMENT


if __name__ == "__main__":
    # for dataset in [WIKITEXT_DATASET]:
    #     for model_id in [GPT2, LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
    #         test_eps(dataset, model_id)
    #         test_delta(dataset, model_id)
    test_eps(WIKITEXT_DATASET, GEMMA_7B)
