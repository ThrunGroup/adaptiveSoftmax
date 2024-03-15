import numpy as np
import torch
from typing import Tuple

from adasoftmax import (
    ada_softmax,
    estimate_mu_hat,
    find_topk_arms,
    approx_sigma,
)
from constants import (
    EPSILON_SCALE,
    DELTA_SCALE,
)

def test_single_normalization(
    A: np.ndarray,
    x: np.ndarray,
    sigma: float,
    beta: float,
    delta: float,
    epsilon: float,
) -> Tuple[bool, int]:
    """
    Tests whether normalization constant S is in the epsilon bounds.
    :returns: correctness, total budget
    """
    true_mu = A @ x
    uniform = np.ones(len(x)) / len(x)
    true_s = np.sum(np.exp(beta * true_mu))

    mu, budget = estimate_mu_hat(
        atoms=A,
        query=x,
        epsilon=epsilon / EPSILON_SCALE,
        delta=delta / DELTA_SCALE,
        sigma=sigma,
        beta=beta,
        dist=uniform,
    )

    s_hat = np.sum(np.exp(beta * mu))
    error = np.abs(s_hat - true_s) / true_s
    is_correct = (error <= epsilon)  
    total_budget = np.sum(budget).item()

    return is_correct, total_budget, 



