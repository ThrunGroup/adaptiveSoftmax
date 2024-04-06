import numpy as np
import torch

from mnl.mnl_utils import get_A_and_x
from mnl.mnl_constants import *
from .test_utils import single_run_adasoftmax


def epsilon_check(dataset):
    A, iterator = get_A_and_x(dataset)
    x, label = next(iterator)
    x = x[0].detach().cpu().numpy()

    in_bounds, error, budget = single_run_adasoftmax(
        A=A,
        x=x,
        k=MNL_TEST_TOPK,
        beta=MNL_TEST_BETA,
        delta=MNL_TEST_DELTA,
        epsilon=MNL_TEST_EPSILON,
        importance=MNL_TEST_IMPORTANCE,
    )
    n, d = A.shape
    return in_bounds, budget, n * d

def delta_check(dataset):
    A, iterator = get_A_and_x(dataset)
    x, label = next(iterator)
    x = x[0].detach().cpu().numpy()
    n, d = A.shape

    total_wrong = 0
    total_budget = 0

    for seed in range(NUM_EXPERIMENTS):
        np.random.seed(seed)
        x, label = next(iterator)
        x = x[0].detach().cpu().numpy()

        # adasoftmax
        in_bounds, error, budget = single_run_adasoftmax(
            A=A,
            x=x,
            k=MNL_TEST_TOPK,
            beta=MNL_TEST_BETA,
            delta=MNL_TEST_DELTA,
            epsilon=MNL_TEST_EPSILON,
            importance=MNL_TEST_IMPORTANCE,
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
    test_eps_mnl_eurosat()
    test_delta_mnl_eurosat()