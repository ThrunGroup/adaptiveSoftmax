import numpy as np
from numba import njit


@njit
def approx_sigma_bound_nb(A, x, n_sigma_sample):
    """
    Add comment here
    """
    # NOTE: We're returning sigma as a sub-gaussian parameter(std of arm pull, not aij*xj)
    elmul = np.multiply(A[:, :n_sigma_sample], x[:n_sigma_sample])
    sigma = np.std(elmul, axis=1)
    return x.shape[0] * np.median(sigma)

@njit
def compute_mip_batch_topk_ver2_warm_nb(
        atoms,
        query,
        sigma,
        delta,
        batch_size=16,
        k=1,
        mu=None,
        budget_vec=None,
):
    """
    does same thing as previous, but instead of doing multiplication between single element of A and x,
    it sequentially slices 'batch_size' elements from left to right, and performs inner product to
    pull an arm.
    """
    dim = len(query)
    n_atoms = len(atoms)
    best_ind = np.empty(n_atoms, dtype=np.int64)
    found_indices_num = 0

    # case where no best-arm identification is needed
    if k == n_atoms:
        return np.arange(n_atoms), 0, mu, budget_vec

    d_used = budget_vec
    n_samples = 0  # instrumentation
    max_index = np.argmax(mu)
    max_mu = mu[max_index]
    # TODO(lukehan): d_used + 1(denominator) should be in the square root

    C_numerator = sigma * 2 * np.log(4 * n_atoms * d_used**2 / delta)
    C_denominator = d_used + 1
    if d_used is not None:
        C = np.sqrt(C_numerator / C_denominator)
    else:
        C = np.zeros(n_atoms)

    if mu is not None:
        solution_mask = (mu + C) >= (max_mu - C[max_index])
    else:
        solution_mask = np.ones(n_atoms, dtype="bool")

    solutions = np.nonzero(solution_mask)[0]
    if len(solutions) <= k:
        best_ind[: len(solutions)] = solutions
        found_indices_num += len(solutions)
        solution_mask = np.logical_not(solution_mask)
        solutions = np.nonzero(solution_mask)[0]

    brute_force_threshold = np.ceil(atoms.shape[0] * 0.05)
    is_not_over_bruteforce_threshold = len(solutions) > brute_force_threshold
    have_not_found_topk = found_indices_num < k
    have_not_sampled_arm_fully = np.max(d_used) < dim - batch_size
    while is_not_over_bruteforce_threshold and have_not_found_topk and have_not_sampled_arm_fully:
        tmp = np.empty(np.sum(solution_mask))
        for i, atom_index in enumerate(solutions):
            v1 = atoms[atom_index, d_used[atom_index]: d_used[atom_index] + batch_size]
            v2 = query[d_used[atom_index]:d_used[atom_index] + batch_size]
            tmp[i] = v1 @ v2

        numerator = np.multiply(d_used[solution_mask], mu[solutions]) + tmp * dim
        denominator = d_used[solution_mask] + batch_size
        mu[solutions] = np.divide(numerator,denominator)
        n_samples += len(solutions) * batch_size

        # TODO(@lukehan): sigma should be outside squareroot
        C_numerator = sigma * 2 * np.log(4 * n_atoms * d_used ** 2 / delta)
        C_denominator = d_used + 1
        C = np.sqrt(np.divide(C_numerator, C_denominator))

        max_index = solutions[np.argmax(mu[solution_mask])]
        max_mu = mu[max_index]
        d_used[solutions] += batch_size
        solution_mask_before = solution_mask
        solution_mask = solution_mask & (mu + C >= max_mu - C[max_index])
        solutions = np.nonzero(solution_mask)[0]

        # TODO(@lukehan): Break this code into a function and consolidate with repeated sections
        if len(solutions) <= k - found_indices_num:
            best_ind[found_indices_num: found_indices_num + len(solutions)] = solutions
            found_indices_num += len(solutions)
            solution_mask = np.logical_xor(solution_mask_before, solution_mask)
            solutions = np.nonzero(solution_mask)[0]

    if found_indices_num < k:
        mu_exact = np.multiply(d_used, mu)
        for i, atom_index in enumerate(solutions):
            mu_exact[i] += atoms[atom_index, d_used[atom_index]:] @ query[d_used[atom_index]:]

        d_used[solutions] = dim
        mu = np.divide(mu_exact, d_used)
        mu_exact_search = mu_exact.copy()

        while found_indices_num < k:
            best_index = np.argmax(mu_exact_search)
            best_ind[found_indices_num] = best_index
            found_indices_num += 1
            mu_exact_search[best_index] = -np.inf

    return best_ind, n_samples, mu, d_used

@njit
def estimate_softmax_normalization_warm_nb(
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
    """
    # TODO: when T0=d, return bruteforce
    true_mu = atoms @ query
    n = atoms.shape[0]
    d = query.shape[0]
    T0 = int(np.ceil(min(np.ceil(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)), d)))

    if verbose:
        print("T0:", T0)

    if T0 == d:
        mu = (atoms @ query).astype(np.float64)
        mu -= np.max(mu)
        S_hat = np.sum(np.exp(mu))

        if verbose:
            # TODO(@lukehan): add beta to errors
            first_order_error = np.sum(np.multiply(np.exp(mu), (true_mu - mu)))
            second_order_error = np.sum(np.multiply(np.exp(mu), (true_mu - mu)**2))
            print(np.full((n,), d).dtype)
            print("first order error:", first_order_error)
            print("second order error:", second_order_error)

        return S_hat, mu, np.full((n,), d).astype(np.int64)

    mu_hat = (d / T0) * (atoms[:, :T0] @ query[:T0])
    C = (2 * sigma ** 2 * np.log(6 * n / delta) / T0) ** 0.5

    mu_hat_aux = (mu_hat - C) * beta
    # TODO(@lukehan): Add sophisticated numericall stability measure
    mu_hat_aux -= np.max(mu_hat_aux)
    mu_hat_exp_alpha = np.exp(mu_hat_aux)
    alpha = mu_hat_exp_alpha / np.sum(mu_hat_exp_alpha)
    mu_hat_exp_gamma = np.exp(mu_hat_aux / 2)
    gamma = mu_hat_exp_gamma / np.sum(mu_hat_exp_gamma)
    # seperate this from constructing alpha and gamma(maximum element subtraction)
    Term1 = 17 * np.log((6 * n) / delta)
    Term2_constant = (16 * (2 ** 0.5) * np.log((6 * n) / delta) * np.sum(mu_hat_exp_gamma)**2) / (epsilon * np.sum(mu_hat_exp_alpha))
    Term3_constant = (16 * np.log(12 / delta)) / (epsilon ** 2)

    if verbose:
        print("ratio:", np.sum(mu_hat_exp_gamma)**2 / (np.sum(mu_hat_exp_alpha)))

    Term2 = gamma * Term2_constant
    Term3 = alpha * Term3_constant

    if verbose:
        print("T1:", Term1)
        print("T2:", Term2)
        print("T3:", Term3)
        print("Sums:", n*Term1, np.sum(Term2), np.sum(Term3))

    #TODO: probably n_samples = max(Term1, Term2, Term3)
    n_samples = (Term2 + Term3) + Term1
    n_samples.astype(np.int64)
    n_samples = np.ceil(np.minimum(beta**2 * sigma**2 * n_samples, d)).astype(np.int64)

    if verbose:
        print("prior scaling:", n_samples)
        #TODO: add beta
        print("estimate n_i:", sigma**2 * n_samples)
        print("T:", np.sum(sigma**2 * n_samples))

    mu_hat_refined = np.empty(n)
    # TODO(@lukehan): how to incorporate beta?
    for i in range(n):
        if n_samples[i] == d:
            mu_hat_refined[i] = atoms[i] @ query
        else:
            mu_hat_refined_aux = d * atoms[i, T0:n_samples[i]] @ query[T0:n_samples[i]]
            mu_hat_refined[i] = (mu_hat[i] * T0 + mu_hat_refined_aux) / max(n_samples[i], 1)

    if verbose:
        first_order_error = np.sum(np.multiply(np.exp(mu_hat_refined), (true_mu - mu_hat_refined)))
        second_order_error = np.sum(np.multiply(np.exp(mu_hat_refined), (true_mu - mu_hat_refined)**2))
        print("first order error:", first_order_error / np.sum(np.exp(beta * true_mu)))
        print("second order error:", second_order_error / np.sum(np.exp(beta * true_mu)))
        print(mu_hat_refined - true_mu)

    # TODO(@lukehan): define seperate vector for mu - max(mu) / mu(untouched)
    mu_hat_refined -= np.max(mu_hat_refined)

    mu_hat_refined_exp = np.exp(mu_hat_refined)
    S_hat = np.sum(mu_hat_refined_exp)
    # Potential hazard: max value of mu_hat for alpha construction
    # and max value of mu hat for S_hat estimation is different, which may
    # result in incorrect sampling or other problems?
    return S_hat, mu_hat_refined, n_samples


# adaSoftmax with warm start
@njit
def ada_softmax_nb(A, x, beta, epsilon, delta, n_sigma_sample, k, verbose=False):
    """
    This calls the other functions --  TODO(@lukehan): fill in
    """
    # TODO(@lukehan):: repleace this with empirical bernstein bound, extend "warm start" to the sigma approximation layer
    sigma = approx_sigma_bound_nb(A, x, n_sigma_sample)
    if verbose:
        print("sigma:", sigma)
        
    # TODO(@lukehan): do NOT compute the S_hat here, just compute estimate on mu
    S_hat, mu_hat, budget_vec = estimate_softmax_normalization_warm_nb(A, x, beta, epsilon / 2, delta / 3, sigma)
    best_index_hat, budget_mip, mu_hat, budget_vec = compute_mip_batch_topk_ver2_warm_nb(A,
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
