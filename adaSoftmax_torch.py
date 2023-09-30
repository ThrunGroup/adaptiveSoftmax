# installing dependencies
import torch
torch.manual_seed(0)
import math
#import numpy as np

def approx_sigma_bound(A, x, n_sigma_sample):
    #NOTE: We're returning sigma as a sub-gaussian parameter(std of arm pull, not aij*xj)
    n_arms = A.shape[0]
    A_subset, x_subset = A[:, :n_sigma_sample], x[:n_sigma_sample]

    elmul = torch.mul(A[:, :n_sigma_sample], x[:n_sigma_sample])

    sigma = torch.empty(n_arms)
    for i in range(n_arms):
        #sigma[i] = np.std(np.array(elmul[i]))
        sigma[i] = torch.std(elmul[i])

    return x.shape[0] * torch.max(sigma).item()

def compute_mip_batch_topk_ver2_warm(atoms, query, sigma, delta, batch_size=16, k=1, mu=None, budget_vec=None):
    """
    does same thing as previous, but instead of doing multiplication between single element of A and x,
    it sequentially slices 'batch_size' elements from left to right, and performs inner product to
    pull an arm.
    """
    # TODO: divide-by-zero occurs on calculating confidence bound(C) when "large" number of sampling happened in normalization estimation.
    # Need to pinpoint the cause.

    dim = len(query)
    n_atoms = len(atoms)

    best_ind = torch.empty(n_atoms)
    found_indices_num = 0

    # case where no best-arm identification is needed
    if k == n_atoms:
        return torch.arange(n_atoms), 0, mu, budget_vec

    d_used = budget_vec
    n_samples = 0  # instrumentation

    mu = mu
    max_index = torch.argmax(mu)
    max_mu = mu[max_index]
    C = torch.div(sigma * torch.sqrt(2 * torch.log(4 * n_atoms * d_used ** 2 / delta)),
                d_used + 1) if d_used is not None else torch.zeros(n_atoms)
    solution_mask = torch.ones(n_atoms).int() & (
                mu + C >= max_mu - C[max_index]) if mu is not None else torch.ones(n_atoms)
    solutions = torch.nonzero(solution_mask, as_tuple=True)[0]
    # topk_indices = np.array([], dtype=np.int64)

    if len(solutions) <= k:
        # topk_indices = np.append(topk_indices, solutions)
        best_ind[found_indices_num: found_indices_num + len(solutions)] = solutions
        found_indices_num += len(solutions)
        solution_mask = torch.logical_not(solution_mask)
        solutions = torch.nonzero(solution_mask, as_tuple=True)[0]
        max_index = solutions[torch.argmax(mu[solution_mask])]
        max_mu = mu[max_index]

    C = torch.divide(sigma * torch.sqrt(2 * torch.log(4 * n_atoms * d_used ** 2 / delta)), d_used + 1)

    solution_mask_before = solution_mask

    brute_force_threshold = math.ceil(atoms.shape[0] * 0.05)

    while (len(solutions) > brute_force_threshold and found_indices_num < k and torch.max(
            d_used) < dim - batch_size):  # TODO: computing max everytime may degrade performance

        tmp = torch.empty(torch.sum(solution_mask))

        for i, atom_index in enumerate(solutions):
            tmp[i] = atoms[atom_index, d_used[atom_index]: d_used[atom_index] + batch_size] @ query[
                                                                                            d_used[atom_index]:
                                                                                            d_used[
                                                                                                atom_index] + batch_size]

        mu[solutions] = torch.div(torch.mul(d_used[solution_mask], mu[solutions]) + tmp * dim,
                                d_used[solution_mask] + batch_size)
        n_samples += len(solutions) * batch_size

        C = torch.div(sigma * torch.sqrt(2 * torch.log(4 * n_atoms * d_used ** 2 / delta)),
                    d_used + 1)  # TODO: update confidence bound. This is when we're sampling one A_iJ * x_J at each round. Can and should be tighter than this -> divide with + batch size somehow?

        max_index = solutions[torch.argmax(mu[solution_mask])]
        max_mu = mu[max_index]

        d_used[solutions] += batch_size

        solution_mask_before = solution_mask
        solution_mask = solution_mask & (mu + C >= max_mu - C[max_index])
        solutions = torch.nonzero(solution_mask, as_tuple=True)[0]

        if len(solutions) <= k - found_indices_num:
            best_ind[found_indices_num: found_indices_num + len(solutions)] = solutions
            found_indices_num += len(solutions)
            solution_mask = torch.logical_xor(solution_mask_before,
                                            solution_mask)  # TODO: Does xor work even in the worst case scenario? or should we revive all candidates?
            solutions = torch.nonzero(solution_mask, as_tuple=True)[0]
            max_index = solutions[torch.argmax(mu[solution_mask])]
            max_mu = mu[max_index]

    # need to check if this is correct?
    if found_indices_num < k:
        mu_exact = torch.mul(d_used[solution_mask], mu[solution_mask])

        tmp = torch.empty(torch.sum(solution_mask))

        for i, atom_index in enumerate(solutions):
            tmp[i] = atoms[atom_index, d_used[atom_index]:] @ query[d_used[atom_index]:]

        mu_exact = (mu_exact + tmp) / dim

        # TODO: is there a way to avoid copy?
        mu_exact_search = mu_exact.detach()

        while found_indices_num < k:
            best_index = torch.argmax(mu_exact_search)
            best_ind[found_indices_num] = best_index
            found_indices_num += 1
            mu_exact_search[best_index] = -float('inf')

        mu[solutions] = mu_exact

        n_samples += torch.sum(dim - d_used[solution_mask])
        d_used[solutions] = dim

    return best_ind.int(), n_samples, mu, d_used

def estimate_softmax_normalization_warm(atoms, query, beta, epsilon, delta, sigma, bruteforce=False):
    #TODO: when T0=d, return bruteforce

    n = atoms.shape[0]
    d = query.shape[0]
    used_samples = 0

    T0 = int(min(math.ceil(17 * beta ** 2 * sigma ** 2 * math.log(6 * n / delta)), d))

    #print("T0:", T0)

    mu_hat = (d / T0) * (atoms[:, :T0] @ query[:T0])
    C = (2 * sigma ** 2 * math.log(6 * n / delta) / T0) ** 0.5

    n_samples = torch.zeros(n) + T0

    #Maybe this is better?
    #mu_hat -= np.min(mu_hat)

    mu_hat_aux = (mu_hat - C) * beta
    mu_hat_aux -= torch.max(mu_hat_aux)
    #mu_hat_aux = (mu_hat - C) * beta
    mu_hat_exp_alpha = torch.exp(mu_hat_aux)
    alpha = mu_hat_exp_alpha / torch.sum(mu_hat_exp_alpha)

    mu_hat_exp_gamma = torch.exp(mu_hat_aux / 2)
    gamma = mu_hat_exp_gamma / torch.sum(mu_hat_exp_gamma)

    #import ipdb; ipdb.set_trace()
    T1 = 17 * math.log((6 * n) / delta) * n
    T2 = (32 * math.log((6 * n) / delta) * n * torch.sum(mu_hat_exp_gamma)**2) / (epsilon * torch.sum(mu_hat_exp_alpha))
    T3 = (16 * math.log(12 / delta)) / (epsilon ** 2)

    T = beta**2 * sigma**2 * (T1 + T2 + T3)

    #Experimental changes
    #normalized_sampling_distribution = (alpha + gamma) / np.sum(alpha + gamma)

    n_samples = torch.ceil(torch.minimum((alpha + gamma) * T + T0, torch.full((n,), d))).int()

    mu_hat_refined_aux = torch.empty(n)

    for i in range(n):
        mu_hat_refined_aux[i] = atoms[i, T0:T0 + n_samples[i]] @ query[T0:T0 + n_samples[i]]

    mu_hat_refined = torch.div(mu_hat * T0 + mu_hat_refined_aux * d, torch.maximum(n_samples, torch.ones(n))) * beta

    #mu_hat_refined -= np.max(mu_hat_refined)

    mu_hat_refined_exp = torch.exp(mu_hat_refined)
    S_hat = torch.sum(mu_hat_refined_exp)

    return S_hat, mu_hat_refined, n_samples.long()

    #Potential hazard: max value of mu_hat for alpha construction and max value of mu hat for S_hat estimation is different, which may result in incorrect sampling or other problems?

# adaSoftmax with warm start
def ada_softmax(A, x, beta, epsilon, delta, n_sigma_sample, k):

    #TODO: repleace this with empirical bernstein bound, extend "warm start" to the sigma approximation layer
    sigma = approx_sigma_bound(A, x, n_sigma_sample)

    #print(sigma)

    S_hat, mu_hat, budget_vec = estimate_softmax_normalization_warm(A, x, beta, epsilon / 2, delta / 3, sigma)

    #print("S estimate:", S_hat)

    #print("denominator budget:", torch.sum(budget_vec))

    best_index_hat, budget_mip, mu_hat, budget_vec = compute_mip_batch_topk_ver2_warm(A, x, sigma, delta / 3,
                                                                                    batch_size=16, k=k, mu=mu_hat,
                                                                                    budget_vec=budget_vec)

    #print("denominator + best arm identification:", torch.sum(budget_vec))

    best_index_hat = best_index_hat[:k]

    n_arm_pull = int(min(
        math.ceil((288 * sigma ** 2 * beta ** 2 * math.log(6 / delta)) / (epsilon ** 2)),
        x.shape[0]
    ))

    #mu_additional = np.empty(k)

    for i in range(best_index_hat.shape[0]):
        #import ipdb; ipdb.set_trace()
        #arm_index = best_index_hat[i]
        arm_index = best_index_hat[i] if best_index_hat.shape[0] > 1 else best_index_hat.item()
        mu_additional = A[arm_index, budget_vec[arm_index]: n_arm_pull] @ x[budget_vec[arm_index]: n_arm_pull]
        mu_hat[arm_index] = (mu_hat[arm_index] * budget_vec[arm_index] + x.shape[0] * mu_additional) / max(budget_vec[arm_index], n_arm_pull, 1)

    #mu_hat[best_index_hat] += np.divide(x.shape[0] * mu_additional, np.maximum(1, n_arm_pull - budget_vec[best_index_hat]))

    #import ipdb; ipdb.set_trace()
    budget_vec[best_index_hat] = torch.maximum(budget_vec[best_index_hat], torch.full((best_index_hat.shape[0],), n_arm_pull).int())

    #print("total_budget:", torch.sum(budget_vec))

    # y_best_hat = np.exp(beta * (mu_best_hat), dtype=np.float64)
    y_best_hat = torch.exp(beta * (mu_hat))
    budget = torch.sum(budget_vec)
    z = y_best_hat / S_hat

    return best_index_hat, z, budget