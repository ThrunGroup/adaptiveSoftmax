import numpy as np
from typing import Tuple, Any

from .utils import (
    approx_sigma,
    get_importance_errors,
    get_fs_errors,
    plot_norm_budgets,
    compare_true_arms,
    create_logs_file,
)

from .constants import (
    BATCH_SIZE,
    TOP_K,
    BETA,
    SAMPLES_FOR_SIGMA,
    DEFAULT_EPSILON,
    DEFAULT_DELTA,
    VERBOSE,

    UNI_CONST,
    F_ORDER_CONST, 
    S_ORDER_CONST,

    DEBUG,
    LOG_PATH,
)


def estimate_mu_hat(
    atoms: np.ndarray,
    query: np.ndarray,
    epsilon: float,
    delta: float,
    sigma: float,
    beta: float,
    verbose: bool = VERBOSE,
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
    log_term = np.log((6 * n) / delta)
  
    # Phase 1: Uniform Sampling to get importance for adaptive sampling
    uni_budget = int(np.ceil(UNI_CONST * 17 * beta**2 * sigma**2 * log_term))
    if verbose:
      print(f"=> uniform budget is {uni_budget * 100 / d:.2f}% of d")

    if uni_budget >= d: # naive comp
        mu = (atoms @ query).astype(np.float64)
        return mu, d * np.ones(n).astype(np.int64)

    # Get mu approximation. Any time mu_hat gets updated, scale by 1/scalar
    # and rescale appropriately with new dimension
    scalar = d / uni_budget
    mu_hat = atoms[:, :uni_budget] @ query[:uni_budget]
    mu_hat *= scalar

    # construct alpha and gamma
    c_interval = np.sqrt(2 * sigma**2 * log_term / uni_budget)
    exp = beta * (mu_hat - c_interval)
    normalized_exp = exp - np.max(exp)  # logsum trick 
    alpha_numer = np.exp(normalized_exp)
    gamma_numer = np.exp(normalized_exp / 2)

    # Phase 2: Adaptive sampling based on importance to find normalization constant.
    uni = UNI_CONST * 17 * log_term # this is the same as unscaled uni_budget
    f_order = 16 * (2 ** 0.5) * log_term * np.sum(gamma_numer) * gamma_numer / (epsilon * np.sum(alpha_numer))
    f_order *= F_ORDER_CONST
    s_order = (16 * np.log(12 / delta)) * alpha_numer / ((epsilon ** 2) * np.sum(alpha_numer))
    s_order *= S_ORDER_CONST

    # get budget "per arms"
    norm_budget = np.maximum(np.maximum(uni, f_order), s_order)
    norm_budget = np.minimum(beta**2 * sigma**2 * norm_budget, d)
    norm_budget = np.ceil(norm_budget).astype(np.int64)

    # one-time sampling for each arm with the budget computed above
    # after sampling appropriately, norm_budget becomes the cumulative budget
    for i in range(n):
        mu_hat[i] /= scalar  # unscale
        scalar_i = d / norm_budget[i]
        mu_hat[i] += atoms[i, uni_budget:norm_budget[i]] @ query[uni_budget:norm_budget[i]]
        mu_hat[i] *= scalar_i  # rescale

    if verbose:
        adaptive_budget = np.sum(norm_budget - uni_budget) / (n * d)
        print(f"=> normalization budget is {adaptive_budget * 100:.2f}% of d across arms")
    
        if DEBUG:   
            true_mu = atoms @ query
            a_error, g_error = get_importance_errors(true_mu, gamma_numer, alpha_numer, beta)
            f_error, s_error = get_fs_errors(true_mu, mu_hat, beta)
            plot_norm_budgets(d, adaptive_budget, a_error, g_error, f_error, s_error)
      
    return mu_hat, norm_budget


def find_topk_arms(
    atoms: np.ndarray,
    query: np.ndarray,
    sigma: float,
    delta: float,
    d_used: np.ndarray,
    mu_hat: np.ndarray,
    batch_size: int = BATCH_SIZE,
    k: int = TOP_K,
    verbose: bool = VERBOSE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to identify the indices of the top k softmax values

    :param atoms: the matrix A in original paper
    :param query: the vector x in original paper
    :param sigma: sub-gaussianity parameter for an arm pull across all arms
    :param delta: probability of being incorrect
    :param mu_hat: mu obtained from the one-time sampling
    :param d_used: number of arm pulls per arm used in the one-time sampling
    :param batch_size: the batch size for each round of UCB & successive elimination
    :param k: the number of indices we want to identify

    :return: top_k indices, mu, number of samples used on each arm
    """
    n, d = atoms.shape
    prev_d_used = d_used.copy()   # this is for debugging

    if k >= n:
        return np.arange(n), mu_hat, d_used

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
        max_index = np.argmax(mu_hat)
        lower_bound = mu_hat[max_index] - c_interval[max_index]
        prev_mask = mask.copy()

        surviving = (mu_hat + c_interval) >= lower_bound
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

        # TODO: is this condition correct? 
        compute_exactly = np.max(d_used[surviving_arms]) > (d - batch_size)
        if compute_exactly or num_found >= k:
            # NOTE: we're not using the terminated variable
            terminated = True
            break

        # update mu approximation with batch_size more samples
        mu_update = np.empty(np.sum(mask))
        for i, arm in enumerate(surviving_arms):
            prev = d_used[arm]
            mu_update[i] = atoms[arm, prev: prev + batch_size] @ query[prev: prev + batch_size]

        # update mu_hat
        scalars = d / d_used[surviving_arms]
        mu_hat[surviving_arms] /= scalars  # unscale

        d_used[surviving_arms] += batch_size  # incremented in place
        mu_hat[surviving_arms] += mu_update
        scalars = d / d_used[surviving_arms]
        mu_hat[surviving_arms] *= scalars  #rescale

    # Brute force computation for the remaining candidates
    if compute_exactly:
        for i, arm in enumerate(surviving_arms):
          used = d_used[arm]
          if used < d:
            scalar = d / d_used[arm]
            mu_hat[arm] /= scalar  # unscale
            mu_hat[arm] += atoms[arm, used:] @ query[used:] 
        
        # At this point, mu_hat is true mu. Need to find the rest of the best arms
        mu_for_search = mu_hat.copy() 
        d_used[surviving_arms] = d
        while num_found < k:
            best_index = np.argmax(mu_hat) 
            best_ind[num_found] = best_index
            mu_for_search[best_index] = -np.inf  # TODO: why are we setting to -inf?
            num_found += 1
    
    if verbose: 
      best_arm_budget = np.sum(d_used) - np.sum(prev_d_used)
      print(f"=> best arm budget is {best_arm_budget * 100 / (n * d):.2f}% of d")
      if DEBUG:
         true_indices, diff = compare_true_arms(atoms @ query, best_ind[:k])

    return np.sort(best_ind[:k]), mu_hat, d_used


def ada_softmax(
    A: np.ndarray,
    x: np.ndarray,
    sigma: float = None,
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
    beta: float = BETA,
    k: int = TOP_K,
    verbose: bool = VERBOSE,
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
    :param verbose: will output how budget is spent on each step

    :return: top-k indices, estimation of softmax value across all indices, and total number of sampled used.
    """
    if DEBUG:
       create_logs_file()
       with open(LOG_PATH, 'w') as f:
            f.write("\n########### starting new experiment ###########\n")
    
    n, d = A.shape
    if sigma is None:
        sigma = approx_sigma(A, x, SAMPLES_FOR_SIGMA)

    # Algorithm 1 in the paper. Denominator (i.e. s_hat) estimation  
    mu_hat, d_used = estimate_mu_hat(
        atoms=A,
        query=x,
        epsilon=epsilon / 2,
        delta=delta / 3,
        sigma=sigma,
        beta=beta,
        verbose=verbose,
    )

    # Best arm identification for y_hat (i.e. numerator) estimation.
    # NOTE: the additional samples used for this process will also be used to update mu
    best_indices, mu_hat, d_used = find_topk_arms(
        atoms=A,
        query=x,
        sigma=sigma,
        delta=delta / 3,
        d_used=d_used,
        batch_size=BATCH_SIZE,
        k=k,
        mu_hat=mu_hat,
        verbose=verbose,
    )

    # Final approximation of mus in the top k indices.
    num_extra = 288 * sigma**2 * beta**2 * np.log(6/delta) / epsilon**2
    n_arm_pull = int(min(d, np.ceil(num_extra)))

    for arm in best_indices:
      used = d_used[arm]
      if used < n_arm_pull:
        mu_additional = d * A[arm, used: n_arm_pull] @ x[used: n_arm_pull]
        mu_hat[arm] = (mu_hat[arm] * used + mu_additional) / n_arm_pull

    if verbose:
      extra_budget = np.sum(d_used[best_indices]).item()
      print(f"=> topk mu budget used is {extra_budget * 100 / (n * d):.2f}% of d across {k} arm(s)")
 
    # Using logsumexp trick for numerical stability
    final_mu_hat = mu_hat - np.max(mu_hat)
    y_hat = np.exp(beta * final_mu_hat)
    s_hat = np.sum(y_hat)
    budget = np.sum(d_used).item()

    return best_indices, y_hat / s_hat, budget 
