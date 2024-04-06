import numpy as np

from adaptive_softmax.utils import approx_sigma
from .test_utils import (
    construct_sanity_example,
    construct_random_example,
    single_run_normalization,
)
from adaptive_softmax.constants import (
    NUM_TESTS,
    NUM_ROWS, 
    NUM_COLS,
    BUDGET_IMPROVEMENT,
    DELTA_SCALE,

    TEST_BETA,
    TEST_DELTA,
    TEST_EPSILON,
    TEST_SEED,
    TEST_IMPORTANCE,
)


def test_epsilon(
    sanity_check: bool = True,
    n: int = NUM_ROWS,
    d: int = NUM_COLS,
    seed: int = TEST_SEED,
) -> None:
    """
    Testing epsilon bounds of normalization constant S
    """
    np.random.seed(seed)
    if sanity_check:
        A, x = construct_sanity_example(n, d)
    else:
        A, x = construct_random_example(n, d, mu=None)

    sigma, _ = approx_sigma(A, x, importance=TEST_IMPORTANCE)
    in_bounds, error, budget = single_run_normalization(
        A=A,
        x=x,
        sigma=sigma,
        beta=TEST_BETA,
        delta=TEST_DELTA,
        epsilon=TEST_EPSILON,
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
    Testing delta bounds of normalization constant S
    """
    np.random.seed(seed)
    total_wrong = 0
    total_budget = 0

    for i in range(num_tests):
        A, x = construct_random_example(n, d, mu=None)
        sigma, _ = approx_sigma(A, x, importance=TEST_IMPORTANCE)

        in_bounds, error, budget = single_run_normalization(
            A=A,
            x=x,
            sigma=sigma,
            beta=TEST_BETA,
            delta=TEST_DELTA,
            epsilon=TEST_EPSILON,
        )
        total_wrong += int(not in_bounds)
        total_budget += budget

    assert (total_wrong / num_tests < TEST_DELTA / DELTA_SCALE)
    assert (total_budget < n * d * num_tests / BUDGET_IMPROVEMENT)



