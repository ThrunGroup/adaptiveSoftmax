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


def test_eps(dataset, model_id):
    llm_constants['model'] = model_id
    in_bounds, budget, naive_budget = epsilon_check(
        dataset, 
        load_llm_matrices, 
        **llm_constants
    )
    assert (in_bounds)
    assert (budget < naive_budget / LLM_TEST_BUDGET_IMPROVEMENT)


def test_delta(dataset, model_id):
    llm_constants['model'] = model_id
    total_wrong, total_budget, naive_budget = delta_check(
        dataset, 
        load_llm_matrices, 
        **llm_constants,
    )
    assert (total_wrong / NUM_EXPERIMENTS < LLM_TEST_DELTA / LLM_DELTA_SCALE)
    assert (total_budget < naive_budget / LLM_TEST_BUDGET_IMPROVEMENT)


if __name__ == "__main__":
    # for dataset in [WIKITEXT_DATASET]:
    #     for model_id in [GPT2, LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
    #         test_eps(dataset, model_id)
    #         test_delta(dataset, model_id)
    test_eps(WIKITEXT_DATASET, GEMMA_7B)






