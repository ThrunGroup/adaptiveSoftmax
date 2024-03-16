import numpy as np

from utils import approx_sigma
from .test_utils import (
    construct_sanity_example,
    construct_random_example,
    single_run_topk,
)
from constants import (
    NUM_TESTS,
    NUM_ROWS, 
    NUM_COLS,
    BUDGET_IMPROVEMENT,
    DELTA_SCALE,

    TEST_BETA,
    TEST_DELTA,
    TEST_EPSILON,
    TEST_SEED,
    TEST_TOPK,
    TEST_IMPORTANCE,
)


def test_epsilon(
    sanity_check: bool = True,
    n: int = NUM_ROWS,
    d: int = NUM_COLS,
    seed: int = TEST_SEED,
) -> None:
    """
    Just keeping naming consistent. There are not error margins here, 
    this is testing whether we retrieve the best k arms. 
    """
    np.random.seed(seed)
    if sanity_check:
        A, x = construct_sanity_example(n, d)
    else:
        A, x = construct_random_example(n, d, mu=None)

    sigma, _ = approx_sigma(A, x, importance=TEST_IMPORTANCE)
    correct, budget = single_run_topk(
        k=TEST_TOPK,
        A=A,
        x=x,
        sigma=sigma,
        delta=TEST_DELTA,
        
        # TODO: what should I set these to?
        starting_mu=None,
        starting_budget=None,
    )

    assert (correct)
    assert (budget < n * d / BUDGET_IMPROVEMENT)


def test_delta(
    num_tests: int = NUM_TESTS,
    n: int = NUM_ROWS,
    d: int = NUM_COLS,
    seed: int = TEST_SEED,
) -> None:
    """
    Testing delta bounds of topk arms
    """
    np.random.seed(seed)
    total_wrong = 0
    total_budget = 0

    for i in range(num_tests):
        A, x = construct_random_example(n, d, mu=None)
        sigma, _ = approx_sigma(A, x, importance=TEST_IMPORTANCE)

        correct, budget = single_run_topk(
            k=TEST_TOPK,
            A=A,
            x=x,
            sigma=sigma,
            delta=TEST_DELTA,
            
            # TODO: what should I set these to?
            starting_mu=None,
            starting_budget=None,
        )
        print(i)
        total_wrong += int(not correct)
        total_budget += budget

    assert (total_wrong / num_tests < TEST_DELTA / DELTA_SCALE)
    assert (total_budget < n * d * num_tests / BUDGET_IMPROVEMENT)



