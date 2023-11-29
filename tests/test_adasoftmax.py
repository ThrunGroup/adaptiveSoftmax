import pytest
import numpy as np
from typing import Tuple

from .constants_t import *
from .utils_t import *
from adaptive_softmax.utils import (
    approx_sigma,
    compare_true_arms,
)
from adaptive_softmax.adasoftmax import (
    ada_softmax,
    estimate_mu_hat,
    find_topk_arms,
)

np.random.seed(42)
@pytest.fixture()
def input():
    # generate ground truth mu randomly
    true_mu = np.ones(NUM_ROWS)
    true_mu = np.random.uniform(1, 100, size=(NUM_ROWS,))
    true_mu /= 10

    # construct A and x that satisfies A @ x = true_mu
    x = np.random.uniform(low=0.94, high=1, size=NUM_COLS)
    Z = np.random.normal(loc=0, scale=1 / NUM_COLS, size=(NUM_ROWS, NUM_COLS))
    A = np.outer(true_mu, x) / np.sum(x ** 2) + Z
    A = A - np.outer(A @ x - true_mu, np.ones(NUM_COLS) / np.sum(x))

    # get the sigma 
    sigma = approx_sigma(A, x, None)
    return A, x, sigma


#############################################################################
# SINGLE instance tests for normalization, best arm identification, and softmax 
# -> this only tests for epsilon error
#############################################################################

def test_normalization(input):
    A, x, sigma = input
    error, budget = compare_normalization(A, x, sigma)
    assert error <= EPSILON / 2  
    # TODO: add contition for budget?
   

def test_topk(input):
    A, x, sigma = input
    indices, true_best, diffs, budget = compare_topk(A, x, sigma)
    assert np.allclose(indices, true_best)
    assert diffs < BEST_ARM_MU_MARGIN
    # TODO: do something for budget?


def test_adasoftmax(input):
    A, x, sigma = input  # calling fixture each time
    empirical_eps, budget = compare_adasoftmax(A, x, sigma)
    assert empirical_eps <= EPSILON

#############################################################################
# NUM_TEST instances for normalization, best arm identification, and softmax. 
# -> this tests for BOTH epsilon and delta error
#############################################################################

def test_normalization_loop(input):
    total_error = 0.0
    num_errors = 0
    total_budget = 0

    for i in range(NUM_TESTS):
        np.random.seed(i)   # TODO: is this setting the seeds correctly?
        A, x, sigma = input
        error, budget = compare_normalization(A, x, sigma)

        total_error += error
        total_budget += budget
        if error >= EPSILON / 2:  # error bound
            num_errors += 1

    # check if the averages are within bounds
    assert (total_error / NUM_TESTS) < EPSILON / 2
    assert (num_errors / NUM_TESTS) < DELTA / 3


def test_topk_loop(input):
    total_error = 0.0
    num_errors = 0
    total_budget = 0

    for i in range(NUM_TESTS):
        np.random.seed(i)
        A, x, sigma = input
        indices, true_best, diffs, budget = compare_topk(A, x, sigma)

        total_error += diffs
        total_budget += budget
        if not np.allclose(indices, true_best):
            num_errors += 1

    # verify that averages are within bounds
    assert diffs/NUM_TESTS < BEST_ARM_MU_MARGIN
    assert num_errors/NUM_TESTS < DELTA / 3


def test_adasoftmax_loop(input):
    total_error = 0.0
    num_errors = 0
    total_budget = 0

    for i in range(NUM_TESTS):
        np.random.seed(i)
        A, x, sigma = input
        empirical_eps, budget = compare_adasoftmax(A, x, sigma)

        total_error += empirical_eps
        total_budget += budget 
        if empirical_eps >= EPSILON / 2:
            num_errors += 1

    # verify that averages are within bounds
    assert empirical_eps/NUM_TESTS < EPSILON / 2
    assert num_errors/NUM_TESTS < DELTA / 3
