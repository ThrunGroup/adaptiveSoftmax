import numpy as np
from numba import njit
from typing import Tuple, List, Any
from constants import exact_computation_epsilon

@njit
def approx_sigma_bound_nb(
    A: np.ndarray,
    x: np.ndarray,
    num_samples: int
) -> float:
    """
    Function to approximate sigma more rigorously. We return the median of the std for the estimation
    (i.e. arm pull) across all arms. But, for now, we assume sigma is passed in as a parameter

    :param A: Matrix A in the original paper
    :param x: Vector x in the original paper
    :param num_samples: number of samples to use for approximating sigma

    :returns: the sigma approximation
    """
    elmul = A[:, :num_samples] * x[:num_samples]
    sigma = np.std(elmul, axis=1)
    # TODO(@ryank, lukehan): Should x.shape[0] be removed here? => Need to check with Tavor
    return x.shape[0] * np.median(sigma)

@njit
def estimate_softmax_normalization(
    atoms: np.ndarray,
    query: np.ndarray,
    beta: float,
    epsilon: float,
    delta: float,
    sigma: float,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    This is Algorithm 1 from the paper (adaApprox).
    Approximates the normalization constant of softmax function (i.e. denoted as S in the original paper) within
    epsilon multiplicative error with probability 1 - delta.

    :param atoms: Matrix A in the original paper
    :param query: Vector x in the original paper
    :param beta: beta in the original paper
    :param epsilon: bound on the multiplicative error of estimation for S
    :param delta: probability of failure of estimation for S
    :param sigma: sub-gaussianity parameter for the arm pull across all arms

    :returns: the approximation for S
    """
    n = atoms.shape[0]
    d = query.shape[0]
    T0 = int(np.ceil(
            min(
                # theoretical complexity
                17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta),
                d
            )
    ))

    # Do exact computation if theoretical complexity isn't less than dimension d

    if T0 >= d:
        mu = (atoms @ query).astype(np.float64)

        if verbose:
            true_mu = atoms @ query
            first_order_error = np.sum(np.exp(beta * mu) * beta * (true_mu - mu))
            second_order_error = np.sum(np.exp(beta * mu) * beta**2 * (true_mu - mu)**2)
            print(np.full((n,), d).dtype)
            print("first order error:", first_order_error)
            print("second order error:", second_order_error)

        return mu, d * np.ones(n).astype(np.int64)

    mu_hat = (d / T0) * (atoms[:, :T0] @ query[:T0])
    C = (2 * sigma ** 2 * np.log(6 * n / delta) / T0) ** 0.5

    # Exponents for alpha and gamma
    mu_hat_aux = (mu_hat - C) * beta

    # Using logsumexp trick for numerical stability
    mu_hat_aux -= np.max(mu_hat_aux)
    mu_hat_exp_alpha = np.exp(mu_hat_aux)
    alpha = mu_hat_exp_alpha / np.sum(mu_hat_exp_alpha)
    mu_hat_exp_gamma = np.exp(mu_hat_aux / 2)
    gamma = mu_hat_exp_gamma / np.sum(mu_hat_exp_gamma)

    ni_term1 = 17 * np.log((6 * n) / delta)
    # TODO(@ryank, lukehan): Break onto several lines
    ni_term2 = gamma * (16 * (2 ** 0.5) * np.log((6 * n) / delta) * np.sum(mu_hat_exp_gamma)**2) / (epsilon * np.sum(mu_hat_exp_alpha))
    ni_term3 = alpha * (16 * np.log(12 / delta)) / (epsilon ** 2)

    n_samples = np.maximum(ni_term1, ni_term2, ni_term3)
    n_samples.astype(np.int64)

    # All terms need to be multiplied by beta^2 * sigma^2
    n_samples = np.ceil(np.minimum(beta**2 * sigma**2 * n_samples, d)).astype(np.int64)

    updated_mu_hat = np.empty(n)
    for i in range(n):
        if n_samples[i] == d:
            updated_mu_hat[i] = atoms[i] @ query
        else:
            # TODO(@ryank, lukehan): Should this be T0:T0 + n_samples[i]? => Discuss with Tavor
            mu_hat_refined_aux = d * atoms[i, T0:n_samples[i]] @ query[T0:n_samples[i]]
            updated_mu_hat[i] = (mu_hat[i] * T0 + mu_hat_refined_aux) / max(n_samples[i], 1)

    if verbose:
        true_mu = atoms @ query
        print("ratio:", np.sum(mu_hat_exp_gamma) ** 2 / (np.sum(mu_hat_exp_alpha)))

        print("T1:", ni_term1)
        print("T2:", ni_term2)
        print("T3:", ni_term3)
        print("Sums:", n * ni_term1, np.sum(ni_term2), np.sum(ni_term3))

        print("prior scaling:", n_samples)
        print("estimate n_i:", beta**2 * sigma ** 2 * n_samples)
        print("T:", np.sum(beta ** sigma ** 2 * n_samples))

        first_order_error = np.sum(np.exp(beta * updated_mu_hat) * (beta * (true_mu - updated_mu_hat)))
        second_order_error = np.sum(np.exp(updated_mu_hat) * (beta**2 * (true_mu - updated_mu_hat)**2))
        print("first order error:", first_order_error / np.sum(np.exp(beta * true_mu)))
        print("second order error:", second_order_error / np.sum(np.exp(beta * true_mu)))
        print(updated_mu_hat - true_mu)

    # Potential hazard: max value of mu_hat for alpha construction
    # and max value of mu hat for S_hat estimation is different, which may
    # result in incorrect sampling or other problems?
    return updated_mu_hat, n_samples

@njit
def find_topk_arms(
    atoms,
    query,
    sigma,
    delta,
    mu_approx,
    d_used,
    batch_size=16,  # TODO(@ryank): Move to constants.py and set batch_size to bigger
    k=1,
):
    """
    Function to identify the top k arms where each arm is ____

    Parameters:
        atoms (ndarray[(n, d), np.float]): the matrix A in original paper
        query (ndarray[(d,), np.float]): the vector x in original paper
        sigma (float): median of the sub-gaussianity parameter for an arm pull across all arms
        delta (float): probability of being incorrect
        mu_approx (ndarray[(n,), np.float]): mu obtained from the normalization approximation
        d_used (ndarray[(n,), np.float]): number of arm pulls per arm
        batch_size (int): the batch size for each round of UCB&successive elimination
        k (int): the number of arms we want to identify

    Returns:
        ndarray[(k,), np.int]: Indices with top-k values in mu
        ndarray[(n,), np.float]: Estimate on mu
        ndarray[(n,), np.int]: Budgets spent on each arm
    """
    dim = len(query)
    n_atoms = len(atoms)

    # case where no best-arm identification is needed
    if k >= n_atoms:
        return np.arange(n_atoms), mu_approx, d_used

    num_found = 0
    best_ind = np.empty(n_atoms, dtype=np.int64)
    mask = np.ones(n_atoms, dtype='bool')   # initially all candidates are valid

    terminated = False
    while not terminated:
        # Run single iteration of successive elimination with statistics shared from normalization step.
        # Namely, the approximation of mu (i.e. mu_approx) and the number of arm pulls made previously (i.e. d_used)
        numer = 2 * np.log(4 * n_atoms * d_used ** 2 / delta)
        denom = d_used + 1
        c_interval = sigma * np.sqrt(numer / denom)

        # update the mask to get the surviving arm indices
        max_index = np.argmax(mu_approx)
        lower_bound = mu_approx[max_index] - c_interval[max_index]

        prev_mask = mask.copy()

        is_surviving_candidate = (mu_approx + c_interval) >= lower_bound
        mask = mask & is_surviving_candidate
        # leaving is as it is because we're not interested in nonzero values, but rather nonzero indices of mask
        surviving_arms = np.nonzero(mask)[0]

        # revive previously eliminated arms if we have less than k surviving arms
        if len(surviving_arms) <= k - num_found:
            best_ind[:len(surviving_arms)] = surviving_arms
            num_found += len(surviving_arms)

            # Performing xor to revive candidates that's eliminated at current round
            mask = np.logical_xor(prev_mask, mask)  # so we don't look at the arms we already found
            surviving_arms = np.nonzero(mask)[0]

        # TODO(@ryank): Comment on why sub-optimality measure is not needed

        compute_exactly = np.max(d_used) > (dim - batch_size)
        if compute_exactly or num_found > k:
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

    # Brute force computation for the remaining candidates
    if compute_exactly:
        curr_mu = d_used * mu_approx
        for i, atom_index in enumerate(surviving_arms):
            curr_mu[i] += atoms[atom_index, d_used[atom_index]:] @ query[d_used[atom_index]:]

        d_used[surviving_arms] = dim
        mu = curr_mu / d_used
        mu_exact_search = curr_mu.copy()

        while num_found < k:
            best_index = np.argmax(mu_exact_search)
            best_ind[num_found] = best_index
            num_found += 1
            mu_exact_search[best_index] = -np.inf

    return best_ind[:k], mu, d_used


# adaSoftmax with warm start
@njit
def ada_softmax(
    A,
    x,
    beta,
    epsilon,
    delta,
    n_sigma_sample,
    k,
    verbose=False,
):
    """
    This is Algorithm 2 from the paper.
    This calls approx_sigma_bound to get an approximation on sigma, the sub-gaussian parameter in the original paper.
    S_hat(estimation on S) is obtained by calling estimate_softmax_normalization.
    Top-k indices are obtained by calling find_topk.
    If additional samples are needed for correct numerator estimation, they're sampled in this function.
    Returns top-k indices, estimation of softmax value across all indices, and total number of sampled used.

    Parameters:
        A (ndarray[(n, d), np.float]): Matrix A in the original paper
        x (ndarray[(d,), np.float]): Vector x in the original paper
        beta (float): Beta in the original paper
        epsilon (float): Multiplicative error bound for softmax value estimation
        delta (float): Probability of failure for softmax value estimation
        n_sigma_sample (int): Number of samples to allocate on sigma approximation
        k (int): Number of top-k softmax values we want to approximate correctly(parameter for find_topk)
        verbose (bool): Indicator for verbosity

    Returns:
        ndarray[(k,), np.int]: Top-k indices for softmax
        ndarray[(n,), np.float]: Estimation for softmax values
        int: Total budget spent for softmax estimation
    """
    sigma = approx_sigma_bound_nb(A, x, n_sigma_sample)
    if verbose:
        print("sigma:", sigma)

    mu_hat, budget_vec = estimate_softmax_normalization(A, x, beta, epsilon / 2, delta / 3, sigma)
    best_index_hat, mu_hat, budget_vec = find_topk_arms(
        A,
        x,
        sigma,
        delta / 3,
        batch_size=16,
        k=k,
        mu=mu_hat,
        budget_vec=budget_vec,
    )

    n_arm_pull = int(min(
        np.ceil((288 * sigma ** 2 * beta ** 2 * np.log(6 / delta)) / (epsilon ** 2)),
        x.shape[0]
    ))

    for arm_index in best_index_hat:
        used_sample = budget_vec[arm_index]
        if used_sample < n_arm_pull:
            mu_additional = x.shape[0] * A[arm_index, used_sample: n_arm_pull] @ x[used_sample: n_arm_pull]
            mu_hat[arm_index] = (mu_hat[arm_index] * used_sample + mu_additional) / max(used_sample, n_arm_pull, 1)

    budget_vec[best_index_hat] = np.maximum(budget_vec[best_index_hat], n_arm_pull)

    #Using logsumexp trick for numerical stability
    mu_hat -= np.max(mu_hat)
    y_best_hat = np.exp(beta * (mu_hat))
    S_hat = np.sum(y_best_hat)
    budget = np.sum(budget_vec)
    z = y_best_hat / S_hat

    return best_index_hat, z, budget

if __name__ == "__main__":
    print(exact_computation_epsilon)
    np.random.seed(777)
