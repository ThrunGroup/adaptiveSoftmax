import numpy as np
from typing import Tuple

from adaptive_softmax.adasoftmax import (
    ada_softmax,
    estimate_mu_hat,
    find_topk_arms,
)
from adaptive_softmax.constants import (
    EPSILON_SCALE,
    DELTA_SCALE,
    TEST_SAMPLES_FOR_SIGMA,
    TEST_MU_LOWER,
    TEST_MU_UPPER,
)


def construct_sanity_example(
    n: int,
    d: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This is the simplest example for which the adaSoftmax algorithm should always work.
    TLDR; A @ x will be a one-hot vector with nonzero elemement = 1

    :returns: A, x
    """
    A = np.zeros((n, d))
    A[0] = 1 / d
    x = np.ones(d)

    return A, x


def construct_random_example(
    n: int,
    d: int,    
    mu: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly construct A, x to get A @ x = mu. 
    If mu isn't given, create random mu. 

    :returns: A, x
    """
    # TODO: better way to construct this??
    if mu is None:
        mu = np.random.uniform(TEST_MU_LOWER, TEST_MU_UPPER, size=(n,))

    x = np.random.uniform(low=0.94, high=1, size=d)  
    Z = np.random.normal(loc=0, scale=1 / d, size=(n, d))
    A = np.outer(mu, x) / np.sum(x**2) + Z
    A = A - np.outer(A @ x - mu, np.ones(d) / np.sum(x))

    return A, x


def single_run_normalization(
    A: np.ndarray,
    x: np.ndarray,
    sigma: float,
    beta: float,
    delta: float,
    epsilon: float,
) -> Tuple[bool, int]:
    """
    Single run of estimate_mu_hat to find normalization constant S in paper (Algo 1, line 3).
    NOTE: the functions names are different from the paper. 
    
    :returns: correctness of S, error, total budget
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
    in_bounds = (error <= epsilon / EPSILON_SCALE)  
    total_budget = np.sum(budget).item()

    return in_bounds, error, total_budget, 


def single_run_topk(
    k: int,
    A: np.ndarray,
    x: np.ndarray,
    delta: float,
    sigma: float,
    starting_mu: np.ndarray = None,
    starting_budget: np.ndarray = None,
) -> Tuple[bool, int]:
    """
    Single run of find_topk_arms to find best arm i* in paper (Algo 1, line 5).
    :returns: correctness of i*, total_budget
    """
    # these are arrays
    true_mu = A @ x
    uniform = np.ones(len(x)) / len(x)
    true_topk = np.sort(np.argpartition(true_mu, -k)[-k:])

    # in full algorithm, these values are obtained from estimate_mu_hat
    if starting_mu is None:
        starting_mu = np.zeros(A.shape[0])   
    if starting_budget is None:
        starting_budget = np.ones(A.shape[0]).astype(np.int64)  # zero arms pulled

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


def single_run_adasoftmax(
    A: np.ndarray,
    x: np.ndarray,
    k: int,
    beta: float,
    delta: float,
    epsilon: float,
    importance: bool,
) -> Tuple[bool, int]:
    """
    Single run of the adaSoftmax algorithm.
    :returns: whether eps is in bounds, error, total budget
    """
    mu = A @ x
    true_s = np.sum(np.exp(beta * mu))
    true_z = np.exp(beta * mu) / true_s
    true_topk = np.sort(np.argpartition(mu, -k)[-k:])

    indices, z_hat, budget = ada_softmax(
        A=A,
        x=x,
        epsilon=epsilon,
        delta=delta,
        samples_for_sigma=TEST_SAMPLES_FOR_SIGMA,
        beta=beta,
        k=k,
        importance=importance,
        verbose=False,
    )
    indices = np.sort(indices)

    # Test results
    error = np.abs(z_hat[indices] - true_z[true_topk]) / true_z[true_topk]
    in_bounds = error <= epsilon  

    return in_bounds, error, budget

