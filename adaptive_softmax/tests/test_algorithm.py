import numpy as np

from .test_utils import (
    construct_sanity_example,
    construct_random_example,
    single_run_adasoftmax
)
from constants import (
    NUM_TESTS,
    NUM_ROWS,
    NUM_COLS,
    TEST_BETA,
    TEST_EPSILON,
    TEST_DELTA,
    TEST_TOPK,
    TEST_SEED,
    BUDGET_IMPROVEMENT,
    TEST_IMPORTANCE,
)

def test_epsilon(
    sanity_check: bool = True,
    n: int = NUM_ROWS,
    d: int = NUM_COLS,
    seed: int = TEST_SEED,
) -> None:
    """
    Testing epsilon bounds of algorithm
    """
    np.random.seed(seed)
    if sanity_check:
        A, x = construct_sanity_example(n, d)
    else:
        A, x = construct_random_example(n, d, mu=None)

    in_bounds, error, budget = single_run_adasoftmax(
        A=A,
        x=x,
        k=TEST_TOPK,
        beta=TEST_BETA,
        delta=TEST_DELTA,
        epsilon=TEST_EPSILON,
        importance=TEST_IMPORTANCE,
    )

    assert (in_bounds)
    assert (budget < n * d / BUDGET_IMPROVEMENT)

    
def test_delta(
    num_tests: int = NUM_TESTS,
    n: int = NUM_ROWS,
    d: int = NUM_COLS,
    seed: int = TEST_SEED,
) -> None:
    """
    Testing delta bounds of algorithm
    """
    np.random.seed(seed)
    total_wrong = 0
    total_budget = 0

    for i in range(num_tests):
        A, x = construct_random_example(n, d, mu=None)

        in_bounds, error, budget = single_run_adasoftmax(
            A=A,
            x=x,
            k=TEST_TOPK,
            beta=TEST_BETA,
            delta=TEST_DELTA,
            epsilon=TEST_EPSILON,
            importance=TEST_IMPORTANCE,
        )
        total_wrong += int(not in_bounds)
        total_budget += budget

    assert (total_wrong / num_tests < TEST_DELTA)
    assert (total_budget < n * d * num_tests / BUDGET_IMPROVEMENT)
    