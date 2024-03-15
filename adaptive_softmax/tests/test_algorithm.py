import numpy as np
from typing import Tuple

from adasoftmax import ada_softmax
from constants import (
    NUM_TESTS,
    NUM_ROWS,
    NUM_COLS,
    TEST_BETA,
    TEST_EPSILON,
    TEST_DELTA,
    TEST_TOPK,
    TEST_SAMPLES_FOR_SIGMA,
    BUDGET_IMPROVEMENT,
)


def single_run(
    A: np.ndarray,
    x: np.ndarray,
    k: int,
    beta: float,
    delta: float,
    epsilon: float,
) -> Tuple[bool, int]:
    # TODO: add the most easiest test cases
    """
    single run of the adaSoftmax algorithm.
    :returns: whether eps is in bounds, total budget
    """
    true_mu = A @ x
    true_s = np.sum(np.exp(beta * true_mu))
    true_z = np.exp(beta * true_mu) / true_s
    true_topk = np.sort(np.argpartition(true_mu, -k)[-k:])
    samples_for_sigma = TEST_SAMPLES_FOR_SIGMA

    indices, z_hat, budget = ada_softmax(
        A=A,
        x=x,
        epsilon=epsilon,
        delta=delta,
        samples_for_sigma=samples_for_sigma,
        beta=beta,
        k=k,
        verbose=False,
    )
    indices = np.sort(indices)

    # Test results
    z_error = np.abs(z_hat[indices] - true_z[true_topk])
    error = np.max(z_error / true_z[true_topk])  # TODO: should we be taking the max?
    in_bounds = error <= epsilon  

    return in_bounds, budget


def test_epsilon() -> None:
    pass


def test_delta(num_tests: int = NUM_TESTS) -> None:
    np.random.seed(42)
    n, d = NUM_ROWS, NUM_COLS
    total_wrong = 0
    total_budget = 0

    for i in range(num_tests):
        # construct A and x such that A @ x = true_mu
        true_mu = np.random.uniform(1, 100, size=(n,)) / n
        x = np.random.uniform(low=0.94, high=1, size=d)
        Z = np.random.normal(loc=0, scale=1 / d, size=(n, d))
        A = np.outer(true_mu, x) / np.sum(x**2) + Z
        A = A - np.outer(A @ x - true_mu, np.ones(d) / np.sum(x))

        is_correct, budget = single_run(
            A=A,
            x=x,
            k=TEST_TOPK,
            beta=TEST_BETA,
            delta=TEST_DELTA,
            epsilon=TEST_EPSILON,
        )
        total_wrong += int(is_correct)
        total_budget += budget

    assert (total_wrong / num_tests < TEST_DELTA)
    assert (total_budget <= n * d / BUDGET_IMPROVEMENT)
    