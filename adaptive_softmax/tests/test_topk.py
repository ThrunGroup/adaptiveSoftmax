import numpy as np
from typing import Tuple

from adasoftmax import (
    find_topk_arms,
)
from constants import (
    DELTA_SCALE,
)

def test_single_topk(
    A: np.ndarray,
    x: np.ndarray,
    n: int,
    k: int,
    delta: float,
    sigma: float,
) -> Tuple[bool, np.ndarray, int]:
    """
    Tests whether top k algorithm returns correct indices
    :returns: correctness, indices, total_budget
    """
    # these are arrays
    true_mu = A @ x
    uniform = np.ones(len(x)) / len(x)
    true_topk = np.sort(np.argpartition(true_mu, -k)[-k:])

    # in full algorithm, these values are obtained from estimate_mu_hat
    starting_mu = true_mu  # TODO: is this the right thing to do?
    starting_budget = np.ones(n).astype(np.int64)  # zero arms pulled

    topk_hat, _, budgets = find_topk_arms(
        atoms=A,
        query=x,
        sigma=sigma,
        delta=delta / DELTA_SCALE,
        mu_hat=starting_mu,
        d_used=starting_budget,
        k=k,
        dist=uniform,
    )

    is_correct = np.allclose(np.sort(topk_hat), true_topk)
    total_budget = np.sum(budgets).item()
    return is_correct, total_budget
