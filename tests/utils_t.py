import numpy as np
from typing import Tuple
from .constants_t import *
from adaptive_softmax.utils import compare_true_arms
from adaptive_softmax.adasoftmax import (
    ada_softmax,
    estimate_mu_hat,
    find_topk_arms,
)

def compare_normalization(
    A: np.ndarray,
    x: np.ndarray,
    sigma: float,
):
    # Estimate the normalization constant using the algorithm
    mu_hat, budget = estimate_mu_hat(
        atoms=A,
        query=x,
        sigma=sigma,
        epsilon=EPSILON / 2,
        delta=DELTA / 3,
        beta=BETA,
    )
    true_norm = np.sum(np.exp(BETA * A@x))
    norm_hat = np.sum(np.exp(BETA * mu_hat))
    error = np.abs(norm_hat - true_norm) / true_norm
    return error, budget

def compare_topk(
    A: np.ndarray,
    x: np.ndarray,
    sigma: float,
):
    start_mu = A[:, 0] * x[0]   
    start_budget = np.ones(A.shape[0]).astype(np.int64)
    best_arms, _, budget = find_topk_arms(
        atoms=A,
        query=x,
        sigma=sigma,
        delta=DELTA / 3,
        mu_hat=start_mu,
        d_used=start_budget,
        k=TOP_K,
    )

    true_best, diffs = compare_true_arms(A@x, best_arms)
    return best_arms, true_best, diffs, budget

def compare_adasoftmax(
    A: np.ndarray,
    x: np.ndarray,
    sigma: float,
):
    mu = A @ x
    true_softmax = np.exp(BETA * mu) / np.sum(np.exp(BETA * mu)) 
    indices, softmax, budget = ada_softmax(
        A=A,
        x=x,
        sigma=sigma,
        epsilon=EPSILON,
        delta=DELTA,
        beta=BETA,
        k=TOP_K,
        verbose=False,
    )
    true_indices, _ = compare_true_arms(mu, indices)
    relevant_softmax_vals = true_softmax[true_indices]
    diff = np.abs(softmax[indices] - relevant_softmax_vals)
    empirical_eps = np.max(diff / relevant_softmax_vals)

    return empirical_eps, budget