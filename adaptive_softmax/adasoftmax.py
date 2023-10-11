import numpy as np
from numba import njit

# TODO(@lukehan, @ryank): add return typehints throughout
# TODO(@lukehan, @ryank): add return docstrings throughout
# TODO(@lukehan, @ryank): add param typehints throughout

@njit
def approx_sigma_bound_nb(A, x, num_samples):
    """
    Function to approximate sigma more rigorously. For now, we assume sigma is passed in as a parameter
    :param A:
    :param x:
    :param num_samples:
    """
    elmul = A[:, :num_samples] * x[:num_samples]
    sigma = np.std(elmul, axis=1)
    # TODO(@ryank, lukehan): Should x.shape[0] be removed here?
    return x.shape[0] * np.median(sigma)

@njit
def estimate_softmax_normalization(
    atoms,
    query,
    beta,
    epsilon,
    delta,
    sigma,
    verbose=False,
):
    """
    What does this do?

    This is Algorithm 1 from the paper (adaApprox)
    """
    # TODO: when T0=d, return bruteforce

    # TODO(@ryank, lukehan): Brute force computation should not be done?
    true_mu = atoms @ query

    n = atoms.shape[0]
    d = query.shape[0]
    T0 = int(np.ceil(min(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta), d)))

    if verbose:
        print("T0:", T0)

    if T0 == d:
        # TODO(@ryank, lukehan): Add a comment that we're doing exact computation
        mu = (atoms @ query).astype(np.float64)
        # TODO(@ryank, lukehan): Add a comment that we're logsumexp trick
        mu -= np.max(mu)
        S_hat = np.sum(np.exp(mu))

        if verbose:
            # TODO(@lukehan): add beta to errors
            first_order_error = np.sum(np.multiply(np.exp(mu), (true_mu - mu)))
            second_order_error = np.sum(np.multiply(np.exp(mu), (true_mu - mu)**2))
            print(np.full((n,), d).dtype)
            print("first order error:", first_order_error)
            print("second order error:", second_order_error)

        return S_hat, mu, d * np.ones(n).astype(np.int64)

    mu_hat = (d / T0) * (atoms[:, :T0] @ query[:T0])
    C = (2 * sigma ** 2 * np.log(6 * n / delta) / T0) ** 0.5

    # Exponents for alpha and gamma
    mu_hat_aux = (mu_hat - C) * beta

    # TODO(@lukehan): Add sophisticated numerical stability measure
    # TODO(@ryank, lukehan): Add a comment that we're logsumexp trick
    mu_hat_aux -= np.max(mu_hat_aux)
    mu_hat_exp_alpha = np.exp(mu_hat_aux)
    alpha = mu_hat_exp_alpha / np.sum(mu_hat_exp_alpha)
    mu_hat_exp_gamma = np.exp(mu_hat_aux / 2)
    gamma = mu_hat_exp_gamma / np.sum(mu_hat_exp_gamma)

    # seperate this from constructing alpha and gamma(maximum element subtraction)
    # TODO(@ryank, lukehan): Call these ni_term1 etc

    Term1 = 17 * np.log((6 * n) / delta)
    # TODO(@ryank, lukehan): Break onto several lines
    ni_term2 = gamma * (16 * (2 ** 0.5) * np.log((6 * n) / delta) * np.sum(mu_hat_exp_gamma)**2) / (epsilon * np.sum(mu_hat_exp_alpha))
    ni_term3 = alpha * (16 * np.log(12 / delta)) / (epsilon ** 2)

    # TODO(@ryank, lukehan): probably n_samples = max(Term1, Term2, Term3)
    n_samples = (ni_term2 + ni_term3) + Term1
    n_samples.astype(np.int64)

    # All terms need to be multiplied by beta^2 * sigma^2
    n_samples = np.ceil(np.minimum(beta**2 * sigma**2 * n_samples, d)).astype(np.int64)

    updated_mu_hat = np.empty(n)
    # TODO(@lukehan): how to incorporate beta?
    for i in range(n):
        if n_samples[i] == d:
            updated_mu_hat[i] = atoms[i] @ query
        else:
            # TODO(@ryank, lukehan): Should this be T0:T0 + n_samples[i]?
            mu_hat_refined_aux = d * atoms[i, T0:n_samples[i]] @ query[T0:n_samples[i]]
            updated_mu_hat[i] = (mu_hat[i] * T0 + mu_hat_refined_aux) / max(n_samples[i], 1)

    if verbose:
        print("ratio:", np.sum(mu_hat_exp_gamma) ** 2 / (np.sum(mu_hat_exp_alpha)))

        print("T1:", Term1)
        print("T2:", ni_term2)
        print("T3:", ni_term3)
        print("Sums:", n * Term1, np.sum(ni_term2), np.sum(ni_term3))

        print("prior scaling:", n_samples)
        # TODO(@ryank, lukehan): add beta
        print("estimate n_i:", sigma ** 2 * n_samples)
        print("T:", np.sum(sigma ** 2 * n_samples))

        first_order_error = np.sum(np.multiply(np.exp(updated_mu_hat), (true_mu - updated_mu_hat)))
        second_order_error = np.sum(np.multiply(np.exp(updated_mu_hat), (true_mu - updated_mu_hat)**2))
        print("first order error:", first_order_error / np.sum(np.exp(beta * true_mu)))
        print("second order error:", second_order_error / np.sum(np.exp(beta * true_mu)))
        print(updated_mu_hat - true_mu)

    # TODO(@lukehan): define seperate vector for mu - max(mu) / mu(untouched)
    updated_mu_hat -= np.max(updated_mu_hat)
    mu_hat_refined_exp = np.exp(updated_mu_hat)
    S_hat = np.sum(mu_hat_refined_exp)

    # Potential hazard: max value of mu_hat for alpha construction
    # and max value of mu hat for S_hat estimation is different, which may
    # result in incorrect sampling or other problems?
    return S_hat, updated_mu_hat, n_samples

@njit
def find_topk_arms(
    atoms,
    query,
    sigma,
    delta,
    mu_approx,
    d_used,
    batch_size=16,  # TODO(@ryank, lukehan): Move to constants.py and set batch_size to bigger
    k=1,
):
    """
    Function to identify the top k arms where each arm is ____
    :param atoms: the matrix A in original paper
    :param query: the vector x in original paper
    :param sigma: the sub-gaussianity parameter for an arm pull
    :param delta: probability of being incorrect
    :param mu_approx: mu obtained from the normalization approximation
    :param d_used: number of arm pulls per arm
    :param batch_size: the batch size
    :param k: the number of arms we want to identify
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
        # TODO(lukehan): d_used + 1(denominator) should be in the square root
        numer = 2 * np.log(4 * n_atoms * d_used ** 2 / delta)
        denom = d_used + 1
        c_interval = sigma * np.sqrt(numer / denom)

        # update the mask to get the surviving arm indices
        max_index = np.argmax(mu_approx)
        lower_bound = mu_approx[max_index] - c_interval[max_index]

        # TODO(@ryank, lukehan): Make this a copy
        prev_mask = mask

        # # TODO(@ryank, lukehan): Break second condition into new variable
        mask = mask & (mu_approx + c_interval) >= lower_bound
        # TODO(@ryank, lukehan): (Throughout) Change out np.nonzero: https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html
        surviving_arms = np.nonzero(mask)[0]    # candidates for a single iteration

        # revive previously eliminated arms if we have less than k surviving arms
        if len(surviving_arms) <= k - num_found:
            best_ind[:len(surviving_arms)] = surviving_arms
            num_found += len(surviving_arms)

            # TODO(@ryank, lukehan): Not one or the other (negative or)
            mask = np.logical_xor(prev_mask, mask)  # so we don't look at the arms we already found
            surviving_arms = np.nonzero(mask)[0]

        # compute exactly if there's only 5% of the original arms left, or we've already sampled past d
        # TODO(@ryank, lukehan): Move 0.05 to constants.py -- no magic numbers

        # # TODO(@ryank, lukehan): This is wrong. We should only be computing exactly if mu_hats are epsilon different, not epsilon number of arms
        threshold = np.ceil(atoms.shape[0] * 0.05)

        compute_exactly = len(surviving_arms) < threshold or np.max(d_used) > (dim - batch_size)
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
    # TODO: need to fix for the case where np.max(d_used) > (dim - batch_size)
    if compute_exactly:
        curr_mu = d_used * mu_approx
        for i, atom_index in enumerate(surviving_arms):
            curr_mu[i] += atoms[atom_index, d_used[atom_index]:] @ query[d_used[atom_index]:]

        d_used[surviving_arms] = dim
        # TODO(@ryank, lukehan): Fix np.divide
        mu = np.divide(curr_mu, d_used)
        mu_exact_search = curr_mu.copy()

        while num_found < k:
            best_index = np.argmax(mu_exact_search)
            best_ind[num_found] = best_index
            num_found += 1
            mu_exact_search[best_index] = -np.inf

    return best_ind[:k], mu, d_used


# adaSoftmax with warm start
# TODO(@ryank, lukehan): Arguments one per line
@njit
def ada_softmax(A, x, beta, epsilon, delta, n_sigma_sample, k, verbose=False):
    """
    This calls the other functions --  TODO(@lukehan): fill in

    This is Algorithm 2 from the paper

    """
    # TODO(@lukehan):: repleace this with empirical bernstein bound, extend "warm start" to the sigma approximation layer
    sigma = approx_sigma_bound_nb(A, x, n_sigma_sample)
    if verbose:
        print("sigma:", sigma)
        
    # TODO(@lukehan): do NOT compute the S_hat here, just compute estimate on mu
    S_hat, mu_hat, budget_vec = estimate_softmax_normalization(A, x, beta, epsilon / 2, delta / 3, sigma)
    best_index_hat, budget_mip, mu_hat, budget_vec = find_topk_arms(
        A,
        x,
        sigma,
        delta / 3,
        batch_size=16,
        k=k,
        mu=mu_hat,
        budget_vec=budget_vec,
    )
    best_index_hat = best_index_hat[:k]

    n_arm_pull = int(min(
        np.ceil((288 * sigma ** 2 * beta ** 2 * np.log(6 / delta)) / (epsilon ** 2)),
        x.shape[0]
    ))

    # TODO(@lukehan): add an if statement to not sample if no additional budget is needed
    for arm_index in best_index_hat:
        # TODO(@lukehan): Make this more readable
        mu_additional = x.shape[0] * A[arm_index, budget_vec[arm_index]: n_arm_pull] @ x[budget_vec[arm_index]: n_arm_pull]
        mu_hat[arm_index] = (mu_hat[arm_index] * budget_vec[arm_index] + mu_additional) / max(budget_vec[arm_index], n_arm_pull, 1)

    budget_vec[best_index_hat] = np.maximum(budget_vec[best_index_hat], n_arm_pull)

    # TODO(@lukehan): use computational trick here
    y_best_hat = np.exp(beta * (mu_hat))
    budget = np.sum(budget_vec)
    z = y_best_hat / S_hat

    return best_index_hat, z, budget

if __name__ == "__main__":
    np.random.seed(777)
