import numpy as np
import torch

from mnl.mnl_utils import get_A_and_x
from mnl.mnl_constants import *
from .test_utils import single_run_adasoftmax

def test_epsilon_mnl():
    for dataset in [MNIST, EUROSAT]:
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
        import ipdb; ipdb.set_trace()
        assert (in_bounds)
        assert (budget < n * d / MNL_TEST_BUDGET_IMPROVEMENT)

def test_delta_mnl():
    for dataset in [MNIST, EUROSAT]:
        A, iterator = get_A_and_x(dataset)
        x, label = next(iterator)
        x = x[0].detach().cpu().numpy()
        n, d = A.shape

        total_wrong = 0
        total_budget = 0
        for seed in range(NUM_EXPERIMENTS):
            np.random.seed(seed)
            x, label = next(iterator)
            x= x[0].detach().cpu().numpy()

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

        assert (total_wrong / NUM_EXPERIMENTS < MNL_TEST_DELTA / MNL_DELTA_SCALE)
        assert (total_budget < n * d * NUM_EXPERIMENTS / MNL_TEST_BUDGET_IMPROVEMENT)


if __name__ == "__main__":
    test_epsilon_mnl()