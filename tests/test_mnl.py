import numpy as np

from adaptive_softmax.sftm import SFTM
from mnl.mnl_utils import load_A_and_xs
from mnl.mnl_constants import *
from .test_utils import single_run_adasoftmax

def epsilon_check(dataset):
    A, xs = load_A_and_xs(dataset)
    n, d = A.shape
    sftm = SFTM(
        A, 
        multiplicative_error=MNL_TEST_EPSILON, 
        failure_probability=MNL_TEST_DELTA, 
        temperature=MNL_TEST_BETA, 
        query_importance_sampling=MNL_TEST_IMPORTANCE
    )
    in_bounds, error, budget = single_run_adasoftmax(
        sftm=sftm,
        x=xs[0],
        k=MNL_TEST_TOPK,
    )
    return in_bounds, budget, n * d

def delta_check(dataset):
    A, xs = load_A_and_xs(dataset)
    sftm = SFTM(
        A, 
        multiplicative_error=MNL_TEST_EPSILON, 
        failure_probability=MNL_TEST_DELTA, 
        temperature=MNL_TEST_BETA, 
        query_importance_sampling=MNL_TEST_IMPORTANCE
    )

    n, d = A.shape
    total_wrong = 0
    total_budget = 0
    for i in range(min(xs.shape[0], NUM_EXPERIMENTS)):
        np.random.seed(i)

        # adasoftmax
        in_bounds, error, budget = single_run_adasoftmax(
            sftm=SFTM,
            x=xs[i],
            k=MNL_TEST_TOPK,
        )
        total_wrong += int(not in_bounds)
        total_budget += budget
    return total_wrong, total_budget, n * d * NUM_EXPERIMENTS

def test_eps_mnl_mnist():
    in_bounds, budget, naive_budget = epsilon_check(MNIST)
    assert (in_bounds)
    assert (budget < naive_budget / MNL_TEST_BUDGET_IMPROVEMENT)

def test_delta_mnl_mnist():
    total_wrong, total_budget, naive_budget = delta_check(MNIST)
    assert (total_wrong / NUM_EXPERIMENTS < MNL_TEST_DELTA / MNL_DELTA_SCALE)
    assert (total_budget < naive_budget / MNL_TEST_BUDGET_IMPROVEMENT)

def test_eps_mnl_eurosat():
    in_bounds, budget, naive_budget = epsilon_check(EUROSAT)
    assert (in_bounds)
    assert (budget < naive_budget / MNL_TEST_BUDGET_IMPROVEMENT)

def test_delta_mnl_eurosat():
    total_wrong, total_budget, naive_budget = delta_check(EUROSAT)
    assert (total_wrong / NUM_EXPERIMENTS < MNL_TEST_DELTA / MNL_DELTA_SCALE)
    assert (total_budget < naive_budget / MNL_TEST_BUDGET_IMPROVEMENT)


if __name__ == "__main__":
    test_eps_mnl_mnist()