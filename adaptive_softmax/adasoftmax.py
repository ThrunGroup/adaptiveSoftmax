import numpy as np
import matplotlib.pyplot as plt

# For approximate_mu
import torch

from numba import njit
from typing import Tuple, List, Any
from constants import (
    BATCH_SIZE,
    TOP_K,
    BETA,
    PROFILE,
    OPTIMIZE_CONSTANTS,
    RETURN_STAGE_BUDGETS,
    DEFAULT_EPSILON,
    DEFAULT_DELTA
)

def precompute_mu(
    A: np.ndarray,
    x: np.ndarray,
):
    num_arms = A.shape[0]
    outlier_frequency = np.zeros(num_arms)

    # TODO: Seems like a bad practice to define function in a function. Better way?
    def find_outliers(A_row):
        """
        Auxilary function to feed in to the apply_along_axis function.
        Find the outlier in A_row*x(element-wise multiplication), and updates the outlier frequency array.

        :param A_row: row of matrix A
        :param x: vector x
        :param outlier_frequency_array: frequency of the indices being an outlier.
        :return: None
        """
        dim = x.shape[0]

        elmul = A_row * x
        num_bins = int(dim ** 0.5)

        freq, edges = np.histogram(elmul[i], bins=num_bins)

        # This is the bin centered around the mean
        most_frequent_bin_index = np.argmax(np.array(freq))
        lower_bound = edges[most_frequent_bin_index]
        upper_bound = edges[most_frequent_bin_index + 1]

        outlier_indicator = np.logical_or(elmul[i] <= lower_bound, elmul[i] >= upper_bound)
        outlier_indices = np.nonzero(outlier_indicator)[0]

        outlier_frequency[outlier_indices] += 1

    np.apply_along_axis(find_outliers, axis=1, arr=A)


    # For profiling purpose
    top_5_outlier_indices = torch.topk(torch.from_numpy(outlier_frequency), 5).indices

    print("top-5 frequent outliers and frequency")
    for i in range(5):
        outlier_index = top_5_outlier_indices[i]
        print("column index j:", outlier_index)
        print("frequency:", outlier_frequency[outlier_index])


    # TODO: find the j's that correspond to the outlier nonzero bins
    print(f"dimensions are {n_arms, dim}\n")
    print(f"num bins {num_bins}\n")
    print(f"freq in bins: {freq}\n")
    print(f"bin edges: {edges}\n")
    print(f"scaled sigma is: {scaled_sigma}\n")


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
    if num_samples is None:
        num_samples = dim

    elmul = A[:, :num_samples] * x[:num_samples]
    sigma = np.std(elmul, axis=1)

    # we need to find index explicitly for debugging purposes
    median_i = np.argsort(sigma)[len(sigma) // 2]
    scaled_sigma = dim * sigma[median_i]

    if verbose:
        num_bins = int(dim ** 0.5)
        freq, edges, _ = plt.hist(elmul[median_i], bins=num_bins)

        # TODO: find the j's that correspond to the outlier nonzero bins
        print(f"dimensions are {n_arms, dim}\n")
        print(f"num bins {num_bins}\n")
        print(f"freq in bins: {freq}\n")
        print(f"bin edges: {edges}\n")
        print(f"scaled sigma is: {scaled_sigma}\n")

    return scaled_sigma


def estimate_mu_hat(
    atoms: np.ndarray,
    query: np.ndarray,
    epsilon: float,
    delta: float,
    sigma: float,
    beta: float = BETA,
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

    :returns: the approximation for mu_hat
    """

    # TODO(@lukehan): Tune these constants, move to a better position for readability
    c_empirical_t0 = 1
    c_empirical_t2 = 1
    c_empirical_t3 = 1

    if OPTIMIZE_CONSTANTS:
        c_empirical_t0 = 0.15
        c_empirical_t2 = 4e-2
        c_empirical_t3 = 1e-3

    n = atoms.shape[0]
    d = query.shape[0]
    T0_original = c_empirical_t0 * 17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)
    T0 = int(np.ceil(
            min(
                # theoretical complexity
                T0_original,
                d
            )
    ).item())

    # Do exact computation if theoretical complexity isn't less than dimension d
    if T0 >= d:
        mu = (atoms @ query).astype(np.float64)

        if verbose:
            true_mu = atoms @ query
            first_order_error = np.sum(np.exp(beta * mu) * beta * (true_mu - mu))
            second_order_error = np.sum(np.exp(beta * mu) * beta**2 * (true_mu - mu)**2)
            print("Exact computation because T0 > d")
            print("T0 was:", T0_original)
            print("first order error:", first_order_error)
            print("second order error:", second_order_error)

        return mu, d * np.ones(n).astype(np.int64)

    # compute variables for lines 2-5 of Algorithm 1
    mu_hat = (atoms[:, :T0] @ query[:T0]) * (d / T0)  # scale to get unbiased estimator
    c_interval = np.sqrt(
        2 * sigma ** 2 * np.log(6 * n / delta) / T0
    )
    exp = beta * (mu_hat - c_interval)
    normalized_exp = exp - np.max(exp)  # logsum trick for numerical stability
    alpha_numer = np.exp(normalized_exp)
    gamma_numer = np.exp(normalized_exp / 2)  # TODO (@lukehan): this is different from paper?

    # Determine the number of total samples to use for each arm. 
    # This means we're taking an extra n_samples - T0 samples to update mu
    log_term = np.log((6 * n) / delta)
    term1 = c_empirical_t0 * 17 * log_term
    term2 = 16 * (2 ** 0.5) * log_term * np.sum(gamma_numer) * gamma_numer / (epsilon * np.sum(alpha_numer))
    term2 *= c_empirical_t2
    term3 = (16 * np.log(12 / delta)) * alpha_numer / ((epsilon ** 2) * np.sum(alpha_numer))
    term3 *= c_empirical_t3
    n_samples = np.maximum(np.maximum(term1, term2), term3).astype(np.int64)
    n_samples = np.ceil(
        np.minimum(beta**2 * sigma**2 * n_samples, d)
    ).astype(np.int64)

    # one-time sampling for each arm with the budget computed above
    updated_mu_hat = np.empty(n)
    for i in range(n):
        if n_samples[i] >= d:
            updated_mu_hat[i] = atoms[i] @ query
        else:
            # TODO: Change mu_
            mu_approx = atoms[i, T0:n_samples[i]] @ query[T0:n_samples[i]] * d
            updated_mu_hat[i] = (mu_hat[i] * T0 + mu_approx) / max(n_samples[i], 1)

    if PROFILE:
        import torch

        # print("T0:", T0, T0_original)

        true_mu = atoms @ query

        normalized_true_mu = true_mu - true_mu.max()

        true_gamma_numer = np.exp((beta * normalized_true_mu) / 2)
        true_gamma = true_gamma_numer / np.sum(true_gamma_numer)
        gamma = gamma_numer / np.sum(gamma_numer)
        gamma_error = gamma / true_gamma
        # print("gamma error top2:", torch.topk(torch.from_numpy(gamma_error), 2).values)

        true_alpha_numer = np.exp(beta * normalized_true_mu)
        true_alpha = true_alpha_numer / np.sum(true_alpha_numer)
        alpha = alpha_numer / np.sum(alpha_numer)
        alpha_error = alpha / true_alpha
        # print("alpha error top2:", torch.topk(torch.from_numpy(alpha_error), 2).values)

        # print("T1:", np.ceil(term1 * beta ** 2 * sigma ** 2))
        # print("T2:", np.ceil(term2 * beta ** 2 * sigma ** 2))
        # print("T3:", np.ceil(term3 * beta ** 2 * sigma ** 2))
        # print("Sums:", n * term1 * beta ** 2 * sigma ** 2, np.sum(np.minimum(term2, d)) * beta ** 2 * sigma ** 2, np.sum(np.minimum(term3, d)) * beta ** 2 * sigma ** 2)

        # print("estimate n_i:", n_samples)
        # print("second phase budget:", np.sum(n_samples))

        first_order_error = np.sum(np.exp(beta * updated_mu_hat) * (beta * (true_mu - updated_mu_hat)))
        second_order_error = np.sum(np.exp(updated_mu_hat) * (beta**2 * (true_mu - updated_mu_hat)**2))
        # print("first order error:", first_order_error / np.sum(np.exp(beta * true_mu)))
        # print("second order error:", second_order_error / np.sum(np.exp(beta * true_mu)))
        # print(updated_mu_hat - true_mu)

        profiling_results = dict()
        profiling_results["gamma_error"] = torch.topk(torch.from_numpy(gamma_error), 2).values
        profiling_results["alpha_error"] = -1 * torch.topk(torch.from_numpy(-1 * alpha_error), 2).values
        profiling_results["T0"] = T0
        profiling_results["top-2 T2"] = torch.topk(torch.from_numpy(np.ceil(term2 * beta ** 2 * sigma ** 2)), 2).values.detach().numpy()
        profiling_results["top-2 T3"] = torch.topk(torch.from_numpy(np.ceil(term3 * beta ** 2 * sigma ** 2)), 2).values.detach().numpy()
        profiling_results["first_order_error"] = first_order_error / np.sum(np.exp(beta * true_mu))
        profiling_results["second_order_error"] = second_order_error / np.sum(np.exp(beta * true_mu))

        S = np.sum(np.exp(true_mu - updated_mu_hat.max()))
        # s_hat =
        # S_error = (S - s_hat) / S

    if PROFILE:
        return updated_mu_hat, n_samples, profiling_results
    else:
        return updated_mu_hat, n_samples


def find_topk_arms(
    atoms: np.ndarray,
    query: np.ndarray,
    sigma: float,
    delta: float,
    mu_approx: np.ndarray,
    d_used: np.ndarray,
    batch_size: int = BATCH_SIZE,
    k: int = TOP_K,
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
    dim = len(query)
    n_atoms = len(atoms)

    # case where no best-arm identification is needed
    if k >= n_atoms:
        return np.arange(n_atoms), mu_approx, d_used

    num_found = 0
    best_ind = np.empty(n_atoms, dtype=np.int64)
    mask = np.ones(n_atoms, dtype='bool')   # initially all candidates are valid

    # Run successive elimination to estimate mu_hat by sampling WITHOUT replacement
    terminated = False
    while not terminated:
        numer = 2 * np.log(4 * n_atoms * d_used ** 2 / delta)
        c_interval = sigma * np.sqrt(numer / (d_used + 1))
        c_interval[d_used == dim] = 0

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
            mask = np.logical_xor(prev_mask, mask)  # revive candidates eliminated at current round
            surviving_arms = np.nonzero(mask)[0]

        compute_exactly = np.max(d_used[surviving_arms]) > (dim - batch_size)
        if compute_exactly or num_found >= k:
            # NOTE: we're not using the terminated variable
            terminated = True
            break

        # update mu approximation with more samples
        sampled_mu_approx = np.empty(np.sum(mask))
        for i, atom_index in enumerate(surviving_arms):
            samples_used = d_used[atom_index]
            v1 = atoms[atom_index, samples_used: samples_used + batch_size]
            v2 = query[samples_used: samples_used + batch_size]
            sampled_mu_approx[i] = v1 @ v2

        # we need to scale the mu approximation accordingly
        numer = mu_approx[surviving_arms] * d_used[mask] + sampled_mu_approx * dim
        mu_approx[surviving_arms] = numer / (d_used[mask] + batch_size)
        d_used[mask] += batch_size

    # Brute force computation for the remaining candidates
    if compute_exactly:
        curr_mu = d_used * mu_approx
        for i, atom_index in enumerate(surviving_arms):
            used = d_used[atom_index]
            if used < dim:
                curr_mu[atom_index] += atoms[atom_index, used:] @ query[used:] * dim

        d_used[surviving_arms] = dim
        mu_approx = curr_mu / d_used  # to maintain consistent naming
        mu_exact_search = curr_mu.copy()

        while num_found < k:
            best_index = np.argmax(mu_exact_search)
            best_ind[num_found] = best_index
            num_found += 1
            mu_exact_search[best_index] = -np.inf

    return best_ind[:k], mu_approx, d_used


# adaSoftmax with warm start

def ada_softmax(
    A: np.ndarray,
    x: np.ndarray,
    samples_for_sigma: int,
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
    beta: float = BETA,
    k: int = TOP_K,
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
    :param verbose: Indicator for verbosity

    :return: top-k indices, estimation of softmax value across all indices, and total number of sampled used.
    """
    sigma = approx_sigma(A=A, x=x, num_samples=None, verbose=True)
    if PROFILE:
        mu_hat, d_used, profiling_results = estimate_mu_hat(
            atoms=A,
            query=x,
            epsilon=epsilon / 2,
            delta=delta / 3,
            sigma=sigma,
            beta=beta,
            verbose=verbose,
        )
    else:
        mu_hat, d_used = estimate_mu_hat(
            atoms=A,
            query=x,
            epsilon=epsilon / 2,
            delta=delta / 3,
            sigma=sigma,
            beta=beta,
            verbose=verbose,
        )

    if PROFILE:
        profiling_results["sigma"] = sigma
        profiling_results["denom budget"] = np.sum(d_used).item()

    if RETURN_STAGE_BUDGETS:
        stage_budgets = dict()
        stage_budgets["denom budget"] = np.sum(d_used).item()

    best_indices, updated_mu_hat, d_used_updated = find_topk_arms(
        atoms=A,
        query=x,
        sigma=sigma,
        delta=delta / 3,
        mu_approx=mu_hat,
        d_used=d_used,
        batch_size=BATCH_SIZE,
        k=k,
    )

    if RETURN_STAGE_BUDGETS:
        stage_budgets["denom+best-arm budget"] = np.sum(d_used_updated).item()

    if PROFILE:
        profiling_results["denom+best-arm budget"] = np.sum(d_used_updated).item()

    # Total samples to use for better approximation of the mu of the top k arms.
    # This means we sample n_arm_pull - used_samples more times.
    n_arm_pull = int(min(
        np.ceil((288 * sigma ** 2 * beta ** 2 * np.log(6 / delta)) / (epsilon ** 2)),
        x.shape[0]
    ))
    for arm_index in best_indices:
        used_sample = d_used_updated[arm_index]
        if used_sample < n_arm_pull:
            mu_additional = x.shape[0] * A[arm_index, used_sample: n_arm_pull] @ x[used_sample: n_arm_pull]
            updated_mu_hat[arm_index] = (updated_mu_hat[arm_index] * used_sample + mu_additional) / n_arm_pull

    d_used_updated[best_indices] = np.maximum(d_used_updated[best_indices], n_arm_pull)

    # Using logsumexp trick for numerical stability
    final_mu_hat = updated_mu_hat - np.max(updated_mu_hat)
    y_hat = np.exp(beta * (final_mu_hat))
    s_hat = np.sum(y_hat)
    budget = np.sum(d_used_updated).item()

    if PROFILE:
        true_mu = A@x
        S = np.sum(np.exp(true_mu - mu_hat.max()))
        S_hat = np.sum(np.exp(mu_hat - mu_hat.max()))
        S_error = (S - S_hat) / S
        profiling_results["denominator error"] = S_error
        # print("S_error:", S_error)
        numer_ground_truth = np.exp(true_mu - updated_mu_hat.max())[best_indices.item()]
        numer_diff = y_hat[best_indices.item()] - numer_ground_truth
        numer_error = numer_diff / numer_ground_truth
        profiling_results["numerator error"] = numer_error
        # print("numerator error:", numer_error)
        profiling_results["final_budget"] = budget

    # print(d_used_updated)
    # TODO: Remove last two return values
    if PROFILE:
        return best_indices, y_hat / s_hat, budget, profiling_results
    elif RETURN_STAGE_BUDGETS:
        return best_indices, y_hat / s_hat, budget, stage_budgets
    else:
        return best_indices, y_hat / s_hat, budget


if __name__ == "__main__":
    # call something?
    pass