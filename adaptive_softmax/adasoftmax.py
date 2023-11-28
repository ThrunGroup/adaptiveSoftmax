from collections import namedtuple
from numba.core.types import parse_integer_signed
import numpy as np
import matplotlib.pyplot as plt
import torch

from hadamard_transform import randomized_hadamard_transform, hadamard_transform
from numba import njit
from typing import Tuple, List, Any
from constants import (
    BATCH_SIZE,
    TOP_K,
    BETA,
    PROFILE,
    PRECOMPUTE,
    OPTIMIZE_CONSTANTS,
    RETURN_STAGE_BUDGETS,
    DEFAULT_EPSILON,
    DEFAULT_DELTA,

    UNI_CONST,
    F_ORDER_CONST, 
    S_ORDER_CONST,

    PLOT_VARIANCE,
    PLOT_BUDGET,
    DEV_RATIO,
)


def approx_sigma(
    A: np.ndarray,
    x: np.ndarray,
    num_samples: Any,
    verbose: bool = False,
) -> float:
    """
    Function to approximate sigma more rigorously. We return the median of the std for the estimation
    (i.e. arm pull) across all arms. But, for now, we assume sigma is passed in as a parameter

    :param A: Matrix A in the original paper
    :param x: Vector x in the original paper
    :param num_samples: number of samples to use for approximating sigma

    :returns: the sigma approximation
    """
    n_arms, dim = A.shape

    # Calculate true sigma unless number of samples to use is explicitly given
    if num_samples is None:
        num_samples = dim

    elmul = A * x
    sigma = np.std(elmul, axis=1)

    # we need to find index explicitly for debugging purposes
    median_i = np.argsort(sigma)[len(sigma) // 2]
    scaled_sigma = dim * sigma[median_i]
    return scaled_sigma

##################### main logic #################

def estimate_mu_hat(
    atoms: np.ndarray,
    query: np.ndarray,
    epsilon: float,
    delta: float,
    sigma: float,
    beta: float,
    true_mu,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This is Algorithm 1 from the paper (adaApprox).
    For numerical stability, we approximate the normalization constant "S" after finding the numerator term of
    the softmax. This function instead returns the mu_hat that will be used to find the normalization constant.

    :param atoms: Matrix A in the original paper
    :param query: Vector x in the original paper
    :param epsilon: bound on the multiplicative error of estimation for S
    :param delta: probability of failure of estimation for S
    :param sigma: sub-gaussianity parameter for the arm pull across all arms
    :param beta: beta in the original paper

    :returns: mu_hat and the current cumulative budget
    """
    n, d = atoms.shape
  
    # Phase 1: Uniform Sampling to get importance for adaptive sampling
    uni_budget = int(np.ceil(UNI_CONST * 17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)))

    if verbose:
      print(f"=> uniform budget is {uni_budget * 100 / d:.2f}% of d")

    if uni_budget >= d: # naive comp
        mu = (atoms @ query).astype(np.float64)
        return mu, d * np.ones(n).astype(np.int64)

    # Get mu approximation. Any time mu_hat gets updated, scale by 1/unbiased_scalar
    # and rescale appropriately with new dimension
    
    unbiased_scalar1 = d / uni_budget
    mu_hat = atoms[:, :uni_budget] @ query[:uni_budget]
    mu_hat *= unbiased_scalar1
    if verbose:
      print(f"  => mu_hat differs by {mu_hat - true_mu}")

    # construct alpha and gamma
    c_interval = np.sqrt(
        2 * sigma ** 2 * np.log(6 * n / delta) / uni_budget
    )
    exp = beta * (mu_hat - c_interval)
    normalized_exp = exp - np.max(exp)  # logsum trick 
    alpha_numer = np.exp(normalized_exp)
    gamma_numer = np.exp(normalized_exp / 2)

    # Phase 2: Adaptive sampling based on importance to find normalization constant.
    log_term = np.log((6 * n) / delta)
    uni = UNI_CONST * 17 * log_term # this is the same as unscaled uni_budget
    f_order = 16 * (2 ** 0.5) * log_term * np.sum(gamma_numer) * gamma_numer / (epsilon * np.sum(alpha_numer))
    f_order *= F_ORDER_CONST
    s_order = (16 * np.log(12 / delta)) * alpha_numer / ((epsilon ** 2) * np.sum(alpha_numer))
    s_order *= S_ORDER_CONST

    # get budget "per arms"
    norm_budget = np.maximum(np.maximum(uni, f_order), s_order)
    norm_budget = np.ceil(
        np.minimum(beta**2 * sigma**2 * norm_budget, d)
    ).astype(np.int64)


    mu_hat_correct = np.zeros(n)
    # one-time sampling for each arm with the budget computed above
    # after sampling appropriately, norm_budget becomes the cumulative budget
    for i in range(n):
        if norm_budget[i] >= d: # naive comp
          mu_hat[i] = atoms[i] @ query

        else:
            mu_hat[i] /= unbiased_scalar1  # unscale
            unbiased_scalar2 = d / norm_budget[i]
            mu_hat[i] += atoms[i, uni_budget:norm_budget[i]] @ query[uni_budget:norm_budget[i]]
            mu_hat[i] *= unbiased_scalar2  # rescale

    if verbose:
      # get budget
      adaptive_budget = np.sum(norm_budget - uni_budget) / (n * d)
      print(f"=> normalization budget is {adaptive_budget * 100:.2f}% of d across arms")

      # get errors
      true_mu = atoms @ query   # TODO: this doesn't account for outliers
      normalized_true_mu = true_mu - true_mu.max()

      true_gamma_numer = np.exp((beta * normalized_true_mu) / 2)
      true_gamma = true_gamma_numer / np.sum(true_gamma_numer)
      gamma = gamma_numer / np.sum(gamma_numer)
      gamma_error = gamma / true_gamma

      true_alpha_numer = np.exp(beta * normalized_true_mu)
      true_alpha = true_alpha_numer / np.sum(true_alpha_numer)
      alpha = alpha_numer / np.sum(alpha_numer)
      alpha_error = alpha / true_alpha

      first_order_error = np.sum(np.exp(beta * mu_hat) * (beta * (true_mu - mu_hat)))
      first_order_error /= np.sum(np.exp(true_mu))

      second_order_error = np.sum(np.exp(mu_hat) * (beta**2 * (true_mu - mu_hat)**2))
      second_order_error /= np.sum(np.exp(true_mu))
      print(f"   => first order error: {first_order_error:.3f}")
      print(f"   => second order error: {second_order_error:.3f}")

    if PLOT_BUDGET:    
      # visualize samples used per arms
      num_bins = 10
      bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
      plt.hist(adaptive_budget/d, bins=bin_edges, edgecolor='black')

      plt.xlabel('ratio of d')
      plt.ylabel('number of arms')
      plt.title('arm pulls for adaptive sampling')
      plt.text(
        0.95, 0.95, 
        f"first order error: {first_order_error:.3f}\n second order error: {second_order_error:.3f}",
        horizontalalignment='right',
        verticalalignment='top',
        transform=plt.gca().transAxes
      )
      plt.savefig(f"normalization_budget.png")
      plt.close()
      
    return mu_hat, norm_budget

def find_topk_arms(
    atoms: np.ndarray,
    query: np.ndarray,
    sigma: float,
    delta: float,
    d_used: np.ndarray,
    mu_approx: np.ndarray,
    batch_size: int = BATCH_SIZE,
    k: int = TOP_K,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to identify the indices of the top k softmax values

    :param atoms: the matrix A in original paper
    :param query: the vector x in original paper
    :param sigma: sub-gaussianity parameter for an arm pull across all arms
    :param delta: probability of being incorrect
    :param mu_approx: mu obtained from the one-time sampling
    :param d_used: number of arm pulls per arm used in the one-time sampling
    :param batch_size: the batch size for each round of UCB & successive elimination
    :param k: the number of indices we want to identify

    :return: top_k indices, mu, number of samples used on each arm
    """
    n, d = atoms.shape
    prev_d_used = d_used.copy()   # this is for debugging

    if k >= n:
        return np.arange(n), mu_approx, d_used

    # initialize variables
    num_found = 0
    best_ind = np.empty(n, dtype=np.int64)
    mask = np.ones(n, dtype='bool')   # initially all candidates are valid

    # Run successive elimination to estimate mu_hat by sampling WITHOUT replacement
    terminated = False
    while not terminated:
        # construct conf interval
        numer = 2 * np.log(4 * n * d_used ** 2 / delta) 
        c_interval = sigma * np.sqrt(numer / (d_used + 1)) 
        c_interval[d_used == d] = 0

        # update the mask to get the surviving arm indices
        max_index = np.argmax(mu_approx)
        lower_bound = mu_approx[max_index] - c_interval[max_index]
        prev_mask = mask.copy()

        surviving = (mu_approx + c_interval) >= lower_bound
        mask = mask & surviving
        surviving_arms = np.nonzero(mask)[0]  # get nonzero indices of mask (this is NOT deprecated)

        # revive previously eliminated arms if we have less than k surviving arms
        if len(surviving_arms) <= k - num_found:
            best_ind[:len(surviving_arms)] = surviving_arms
            num_found += len(surviving_arms)

            # We want to revive only the candidates that have been removed from current rounds
            # this means we want to set surviving_arms to true only when current mask is False,
            # and previous mask is True, and every other mask should be False.
            # XOR operation gives the desired result, hence we're using XOR.
            mask = np.logical_xor(prev_mask, mask)  # revive candidates eliminated at current round
            surviving_arms = np.nonzero(mask)[0]

        compute_exactly = np.max(d_used[surviving_arms]) > (d - batch_size)
        if compute_exactly or num_found >= k:
            # NOTE: we're not using the terminated variable
            terminated = True
            break

        # update mu approximation with more samples
        sampled_mu = np.empty(np.sum(mask))
        for i, atom_index in enumerate(surviving_arms):
            samples_used = d_used[atom_index]
            v1 = atoms[atom_index, samples_used: samples_used + batch_size]
            v2 = query[samples_used: samples_used + batch_size]
            sampled_mu[i] = v1 @ v2

        # we need to scale the mu approximation accordingly
        unbiased1 = d / d_used[surviving_arms]
        mu_approx[surviving_arms] /= unbiased1  # unscale

        d_used[surviving_arms] += batch_size
        mu_approx[surviving_arms] += sampled_mu
        unbiased2 = d / d_used[surviving_arms]
        mu_approx[surviving_arms] *= unbiased2 #rescale

    # Brute force computation for the remaining candidates
    # TODO: MAKE SURE THIS LOGIC IS CORRECT (scaling is likely off)
    if compute_exactly:
        #curr_mu = d_used * mu_approx
        for i, atom_index in enumerate(surviving_arms):
          used = d_used[atom_index]
          if used < d:
            unbiased3 = d / d_used[atom_index]
            mu_approx[atom_index] /= unbiased3  # unscale
            mu_approx[atom_index] += atoms[atom_index, used:] @ query[used:] 

        mu_for_search = mu_approx.copy() # Need a copy of approximation, because we set the value of best arms to -inf(to cover k>1)

        d_used[surviving_arms] = d
        while num_found < k:
            best_index = np.argmax(mu_approx) # at this point, mu_approx is true mu
            best_ind[num_found] = best_index
            num_found += 1
            mu_for_search[best_index] = -np.inf
    
    if verbose: 
      # TODO: represent as histogram
      true_mu = atoms @ query
      best_arm_budget = np.sum(d_used) - np.sum(prev_d_used)
      print(f"=> best arm budget is {best_arm_budget * 100 / (n * d):.2f}% of d")
      print("   => indices: ", (best_ind[:k][0], np.argmax(true_mu)))
      print("   => diff: ", (np.max(true_mu) - true_mu[best_ind[:k]])[0])

    return best_ind[:k], mu_approx, d_used


def ada_softmax(
    A: np.ndarray,
    x: np.ndarray,
    samples_for_sigma: int,
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
    beta: float = BETA,
    k: int = TOP_K,
    precompute: bool = PRECOMPUTE,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    This is Algorithm 2 from the paper.
    This calls approx_sigma_bound to get an approximation on sigma, the sub-gaussian parameter in the original paper.
    The estimation on S is obtained here. Additional samples may be used for the precise estimation for the best k arms.

    :param A: Matrix A in the original paper
    :param x: Vector x in the original paper
    :param epsilon: Multiplicative error bound for softmax value estimation
    :param delta: Probability of failure for softmax value estimation
    :param samples_for_sigma: Number of samples to allocate on sigma approximation
    :param beta: Beta in the original paper
    :param k: Number of top-k softmax values we want to approximate correctly(parameter for find_topk)
    :param precompute: Whether we should precompute the "heavy hitters" i.e. columns of j with large variance
    :param verbose: Indicator for verbosity

    :return: top-k indices, estimation of softmax value across all indices, and total number of sampled used.
    """

    # precompute outliers
    n, d = A.shape
    true_mu = A @ x

    sigma = approx_sigma(
        A=A,
        x=x,
        num_samples=samples_for_sigma,
        verbose=verbose
    )

    # Algorithm 1 in the paper. Denominator (i.e. s_hat) estimation  
    mu_hat, d_used = estimate_mu_hat(
        atoms=A,
        query=x,
        epsilon=epsilon / 2,
        delta=delta / 3,
        sigma=sigma,
        beta=beta,
        true_mu=true_mu,  # for debugging
        verbose=verbose,
    )
    
    if verbose:
      print(f"  => mu_hat differs by {mu_hat - true_mu}")

    # Best arm identification for y_hat (i.e. numerator) estimation.
    # NOTE: the additional samples used for this process will also be used to update mu
    best_indices, mu_hat, d_used_updated = find_topk_arms(
        atoms=A,
        query=x,
        sigma=sigma,
        delta=delta / 3,
        d_used=d_used,
        batch_size=BATCH_SIZE,
        k=k,
        mu_approx=mu_hat,
        verbose=verbose,
    )

    if verbose:
      print(f"  => mu_hat differs by {mu_hat - true_mu}")

    # Total samples to use for better approximation of the mu of the top k arms.
    # This means we sample (n_arm_pull - used_samples) more times and update mu accordingly
    dim = x.shape[0]
    n_arm_pull = int(min(
        np.ceil((288 * sigma ** 2 * beta ** 2 * np.log(6 / delta)) / (epsilon ** 2)),
        dim
    ))
    for arm_index in best_indices:
      used_sample = d_used_updated[arm_index]
      if used_sample < n_arm_pull:
        """
        d_used_updated[arm_index] += n_arm_pull
        unbiased1 = d / used_sample
        mu_hat[arm_index] /= unbiased1  # unscale
        mu_hat[arm_index] += A[arm_index, used_sample: n_arm_pull] @ x[used_sample: n_arm_pull]
        
        unbiased2 = d / d_used_updated[arm_index]
        mu_hat[arm_index] *= unbiased2   # rescale
        """
        mu_additional = d * A[arm_index, used_sample: n_arm_pull] @ x[used_sample: n_arm_pull]
        mu_hat[arm_index] = (mu_hat[arm_index] * used_sample + mu_additional) / n_arm_pull

    # total budget
    budget = np.sum(d_used_updated).item()
    if verbose:
      extra_budget = np.sum(d_used_updated[best_indices]).item()
      print(f"=> topk mu budget used is {extra_budget * 100 / (n * d):.2f}% of d across {k} arm(s)")
      print(f"  => mu_hat diffs by {mu_hat - true_mu}")
    
    # print(mu_hat - true_mu)

    # Using logsumexp trick for numerical stability
    final_mu_hat = mu_hat - np.max(mu_hat)
    y_hat = np.exp(beta * final_mu_hat)
    s_hat = np.sum(y_hat)

    return best_indices, y_hat / s_hat, budget 


if __name__ == "__main__":
    # call something?
    pass
