from .test_utils import epsilon_check, delta_check
from mnl.mnl_utils import load_A_and_xs
from mnl.mnl_constants import (
    MNL_TEST_EPSILON,
    MNL_TEST_DELTA,
    MNL_TEST_BETA,
    MNL_TEST_IMPORTANCE,
    MNL_TEST_TOPK,
    NUM_EXPERIMENTS,
    MNL_TEST_BUDGET_IMPROVEMENT,
    MNL_DELTA_SCALE,
    MNIST,
    EUROSAT,
)

constants = {
    'is_mnl': True,
    'eps': MNL_TEST_EPSILON,
    'delta': MNL_TEST_DELTA,
    'temp': MNL_TEST_BETA,
    'query_importance': MNL_TEST_IMPORTANCE,
    'top_k': MNL_TEST_TOPK,
    'num_experiments': NUM_EXPERIMENTS
}

def test_eps_mnl_mnist():
    in_bounds, budget, naive_budget = epsilon_check(MNIST, load_A_and_xs, **constants)
    assert (in_bounds)
    assert (budget < naive_budget / MNL_TEST_BUDGET_IMPROVEMENT)

def test_delta_mnl_mnist():
    total_wrong, total_budget, naive_budget = delta_check(MNIST, load_A_and_xs, **constants)
    assert (total_wrong / NUM_EXPERIMENTS < MNL_TEST_DELTA / MNL_DELTA_SCALE)
    assert (total_budget < naive_budget / MNL_TEST_BUDGET_IMPROVEMENT)

def test_eps_mnl_eurosat():
    in_bounds, budget, naive_budget = epsilon_check(EUROSAT, load_A_and_xs, **constants)
    assert (in_bounds)
    assert (budget < naive_budget / MNL_TEST_BUDGET_IMPROVEMENT)

def test_delta_mnl_eurosat():
    total_wrong, total_budget, naive_budget = delta_check(EUROSAT, load_A_and_xs, **constants)
    assert (total_wrong / NUM_EXPERIMENTS < MNL_TEST_DELTA / MNL_DELTA_SCALE)
    assert (total_budget < naive_budget / MNL_TEST_BUDGET_IMPROVEMENT)


if __name__ == "__main__":
    test_eps_mnl_mnist()
    test_delta_mnl_mnist()
    #test_eps_mnl_eurosat()
    #test_delta_mnl_eurosat()
