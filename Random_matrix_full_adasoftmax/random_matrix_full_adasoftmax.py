import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import torch
from hadamard_transform import randomized_hadamard_transform, hadamard_transform
np.random.seed(777)

@njit
def approx_sigma_bound_nb(A, x, n_sigma_sample):
    #NOTE: We're returning sigma as a sub-gaussian parameter(std of arm pull, not aij*xj)
    n_arms = A.shape[0]
    A_subset, x_subset = A[:, :n_sigma_sample], x[:n_sigma_sample]

    elmul = np.multiply(A[:, :n_sigma_sample], x[:n_sigma_sample])

    sigma = np.empty(n_arms)
    for i in range(n_arms):
        sigma[i] = np.std(elmul[i])

    #print(sigma)

    return x.shape[0] * np.median(sigma)

@njit
def compute_mip_batch_topk_ver2_warm_nb(atoms, query, sigma, delta, batch_size=16, k=1, mu=None, budget_vec=None):
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

    mu = mu
    max_index = np.argmax(mu)
    max_mu = mu[max_index]
    #d_used + 1(denominator) should be in the square root
    C = np.divide(sigma * np.sqrt(2 * np.log(4 * n_atoms * d_used ** 2 / delta)),
                  d_used + 1) if d_used is not None else np.zeros(n_atoms)

    solution_mask = np.ones(n_atoms, dtype="bool") & (
                mu + C >= max_mu - C[max_index]) if mu is not None else np.ones(n_atoms, dtype="bool")
    solutions = np.nonzero(solution_mask)[0]
    # topk_indices = np.array([], dtype=np.int64)

    if len(solutions) <= k:
        # topk_indices = np.append(topk_indices, solutions)
        best_ind[found_indices_num: found_indices_num + len(solutions)] = solutions
        found_indices_num += len(solutions)
        solution_mask = np.logical_not(solution_mask)
        solutions = np.nonzero(solution_mask)[0]
        max_index = solutions[np.argmax(mu[solution_mask])]
        max_mu = mu[max_index]

    #print(mu)

    C = np.divide(sigma * np.sqrt(2 * np.log(4 * n_atoms * d_used ** 2 / delta)), d_used + 1)

    print(d_used)

    solution_mask_before = solution_mask

    brute_force_threshold = np.ceil(atoms.shape[0] * 0.05)

    while (len(solutions) > brute_force_threshold and found_indices_num < k and np.max(
            d_used) < dim - batch_size):  # TODO: computing max everytime may degrade performance

        #print("flag")

        tmp = np.empty(np.sum(solution_mask))

        for i, atom_index in enumerate(solutions):
            tmp[i] = atoms[atom_index, d_used[atom_index]: d_used[atom_index] + batch_size] @ query[
                                                                                              d_used[atom_index]:
                                                                                              d_used[
                                                                                                  atom_index] + batch_size]

        mu[solutions] = np.divide(np.multiply(d_used[solution_mask], mu[solutions]) + tmp * dim,
                                  d_used[solution_mask] + batch_size)
        n_samples += len(solutions) * batch_size

        C = np.divide(sigma * np.sqrt(2 * np.log(4 * n_atoms * d_used ** 2 / delta)),
                      d_used + 1)  # TODO: update confidence bound. This is when we're sampling one A_iJ * x_J at each round. Can and should be tighter than this -> divide with + batch size somehow?

        max_index = solutions[np.argmax(mu[solution_mask])]
        max_mu = mu[max_index]

        d_used[solutions] += batch_size

        solution_mask_before = solution_mask
        solution_mask = solution_mask & (mu + C >= max_mu - C[max_index])
        solutions = np.nonzero(solution_mask)[0]

        if len(solutions) <= k - found_indices_num:
            best_ind[found_indices_num: found_indices_num + len(solutions)] = solutions
            found_indices_num += len(solutions)
            solution_mask = np.logical_xor(solution_mask_before,
                                            solution_mask)  # TODO: Does xor work even in the worst case scenario? or should we revive all candidates?
            solutions = np.nonzero(solution_mask)[0]
            max_index = solutions[np.argmax(mu[solution_mask])]
            max_mu = mu[max_index]

    # need to check if this is correct?
    if found_indices_num < k:
        mu_exact = np.multiply(d_used, mu)


        for i, atom_index in enumerate(solutions):
            mu_exact[i] += atoms[atom_index, d_used[atom_index]:] @ query[d_used[atom_index]:]

        d_used[solutions] = dim

        mu = np.divide(mu_exact, d_used)

        # TODO: is there a way to avoid copy?
        mu_exact_search = mu_exact.copy()

        while found_indices_num < k:
            best_index = np.argmax(mu_exact_search)
            best_ind[found_indices_num] = best_index
            found_indices_num += 1
            mu_exact_search[best_index] = -np.inf

    return best_ind, n_samples, mu, d_used

@njit
def estimate_softmax_normalization_warm_nb(atoms, query, beta, epsilon, delta, sigma, bruteforce=False):
    #TODO: when T0=d, return bruteforce
    true_mu = atoms @ query

    n = atoms.shape[0]
    d = query.shape[0]
    used_samples = 0

    #TODO: do this on other int functions
    T0 = int(np.ceil(min(np.ceil(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)), d)))
    print("T0:", T0)

    if T0 == d:
        mu = (atoms @ query).astype(np.float64)
        #TODO: add beta to errors
        first_order_error = np.sum(np.multiply(np.exp(mu), (true_mu - mu)))
        second_order_error = np.sum(np.multiply(np.exp(mu), (true_mu - mu)**2))
        mu -= np.max(mu)
        S_hat = np.sum(np.exp(mu))
        print(np.full((n,), d).dtype)
        print("first order error:", first_order_error)
        print("second order error:", second_order_error)
        return S_hat, mu, np.full((n,), d).astype(np.int64)

    mu_hat = (d / T0) * (atoms[:, :T0] @ query[:T0])
    C = (2 * sigma ** 2 * np.log(6 * n / delta) / T0) ** 0.5

    C_true = (2 * sigma ** 2 * np.log(6 * n / delta) / d) ** 0.5

    true_alpha = np.exp(true_mu - C_true) / np.sum(np.exp(true_mu - C_true))
    true_gamma = np.exp((true_mu - C_true) / 2) / np.sum(np.exp((true_mu - C_true) / 2))

    #Maybe this is better?
    #mu_hat -= np.min(mu_hat)

    mu_hat_aux = (mu_hat - C) * beta
    #TODO: Add sophisticated numericall stability measure
    #mu_hat_aux -= np.max(mu_hat_aux)
    mu_hat_exp_alpha = np.exp(mu_hat_aux)
    alpha = mu_hat_exp_alpha / np.sum(mu_hat_exp_alpha)

    mu_hat_exp_gamma = np.exp(mu_hat_aux / 2)
    gamma = mu_hat_exp_gamma / np.sum(mu_hat_exp_gamma)


    #seperate this from constructing alpha and gamma(maximum element subtraction)
    #import ipdb; ipdb.set_trace()
    Term1 = 17 * np.log((6 * n) / delta)
    Term2_constant = (16 * (2 ** 0.5) * np.log((6 * n) / delta) * np.sum(mu_hat_exp_gamma)**2) / (epsilon * np.sum(mu_hat_exp_alpha))
    Term3_constant = (16 * np.log(12 / delta)) / (epsilon ** 2)

    print("ratio:", np.sum(mu_hat_exp_gamma)**2 / (np.sum(mu_hat_exp_alpha)))

    Term2 = gamma * Term2_constant
    Term3 = alpha * Term3_constant

    print("T1:", Term1)
    print("T2:", Term2)
    print("T3:", Term3)
    print("Sums:", n*Term1, np.sum(Term2), np.sum(Term3))

    #probably n_samples = max(Term1, Term2, Term3)
    n_samples = (Term2 + Term3) + Term1
    n_samples.astype(np.int64)
    print("prior scaling:", n_samples)
    #TODO: add beta
    print("estimate n_i:", sigma**2 * n_samples)
    print("T:", np.sum(sigma**2 * n_samples))
    n_samples = np.ceil(np.minimum(beta**2 * sigma**2 * n_samples, d)).astype(np.int64) #TODO: Note that we're ignoring beta here, need to add support later. / Redundant typing?

    #n_samples = np.ceil(np.minimum((alpha + gamma) * T + T0, d)).astype(np.int64)

    mu_hat_refined = np.empty(n)

    #TODO: how to incorporate beta?
    for i in range(n):
        if n_samples[i] == d:
            mu_hat_refined[i] = atoms[i] @ query
        else:
            mu_hat_refined_aux = d * atoms[i, T0:n_samples[i]] @ query[T0:n_samples[i]]
            mu_hat_refined[i] = (mu_hat[i] * T0 + mu_hat_refined_aux) / max(n_samples[i], 1)

    #normalize only on arms that are NOT bruteforces

    first_order_error = np.sum(np.multiply(np.exp(mu_hat_refined), (true_mu - mu_hat_refined)))
    second_order_error = np.sum(np.multiply(np.exp(mu_hat_refined), (true_mu - mu_hat_refined)**2))

    print("first order error:", first_order_error / np.sum(np.exp(beta * true_mu)))
    print("second order error:", second_order_error / np.sum(np.exp(beta * true_mu)))

    print(mu_hat_refined - true_mu)

    #TODO: define seperate vector for mu - max(mu) / mu(untouched)
    mu_hat_refined -= np.max(mu_hat_refined)

    mu_hat_refined_exp = np.exp(mu_hat_refined)
    S_hat = np.sum(mu_hat_refined_exp)

    return S_hat, mu_hat_refined, n_samples

    #Potential hazard: max value of mu_hat for alpha construction and max value of mu hat for S_hat estimation is different, which may result in incorrect sampling or other problems?

# adaSoftmax with warm start
@njit
def ada_softmax_nb(A, x, beta, epsilon, delta, n_sigma_sample, k):

    #TODO: repleace this with empirical bernstein bound, extend "warm start" to the sigma approximation layer
    sigma = approx_sigma_bound_nb(A, x, n_sigma_sample)

    print("sigma:", sigma)
    #sigma = 6.0

    #TODO: do NOT compute the S_hat here, just compute estimate on mu
    S_hat, mu_hat, budget_vec = estimate_softmax_normalization_warm_nb(A, x, beta, epsilon / 2, delta / 3, sigma)

    print(budget_vec)

    mu = A @ x
    mu -= np.max(mu)
    S = np.sum(np.exp(mu))
    print("S error:", S_hat - S)

    #print("S estimate:", S_hat)

    print("denominator budget:", np.sum(budget_vec))

    #print(mu_hat)

    best_index_hat, budget_mip, mu_hat, budget_vec = compute_mip_batch_topk_ver2_warm_nb(A, x, sigma, delta / 3,
                                                                                      batch_size=16, k=k, mu=mu_hat,
                                                                                      budget_vec=budget_vec)

    print("denominator + best arm identification:", np.sum(budget_vec))
    #print(mu_hat)

    best_index_hat = best_index_hat[:k]

    n_arm_pull = int(min(
        np.ceil((288 * sigma ** 2 * beta ** 2 * np.log(6 / delta)) / (epsilon ** 2)),
        x.shape[0]
    ))

    #mu_additional = np.empty(k)

    #TODO: add an if statement to not sample if no additional budget is needed
    for arm_index in best_index_hat:
        mu_additional = x.shape[0] * A[arm_index, budget_vec[arm_index]: n_arm_pull] @ x[budget_vec[arm_index]: n_arm_pull]
        mu_hat[arm_index] = (mu_hat[arm_index] * budget_vec[arm_index] + mu_additional) / max(budget_vec[arm_index], n_arm_pull, 1)

    #mu_hat[best_index_hat] += np.divide(x.shape[0] * mu_additional, np.maximum(1, n_arm_pull - budget_vec[best_index_hat]))

    budget_vec[best_index_hat] = np.maximum(budget_vec[best_index_hat], n_arm_pull)

    #print("total_budget:", np.sum(budget_vec))

    # y_best_hat = np.exp(beta * (mu_best_hat), dtype=np.float64)
    #TODO: use computational trick here
    y_best_hat = np.exp(beta * (mu_hat))
    budget = np.sum(budget_vec)
    z = y_best_hat / S_hat

    #z = z / np.sum(z)

    return best_index_hat, z, budget

n = 10
beta = 1
epsilon = 0.1
delta = 0.01
k = 1
N_EXPERIMENTS = 1

"""
true_mu = np.ones(n)
true_mu[1] = true_mu[1] * c

x = np.random.normal(loc=0.9, scale=0.01, size=d)
Z = np.random.normal(loc=0, scale=1/d, size=(n, d))
A = np.outer(true_mu, x)/np.sum(x**2) + Z
A = A - np.outer(A@x - true_mu, np.ones(d)/np.sum(x))
"""

dimension_list = list()
budget_list = list()

for d in range(350000, 460000, 10000):
  dimension_list.append(d)

  error_sum = 0.0
  wrong_approx_num = 0

  #generate "weight", normalize to same l2 norm across all rows
  """
  A = np.random.normal(loc=0, scale=0.1, size=(n, d))
  A_norm = np.linalg.norm(A, axis=1)

  for i in range(n):
    A[i] = A[i] / (A_norm[i] / 2.4)

  dPad = int(2**np.ceil(np.log2(d)))
  print(f'padded dimension: {dPad}')
  Apad = np.pad(A,((0,0),(0,dPad-d)),'constant',constant_values=0)

  prng = torch.Generator(device='cpu')
  hadamard_seed = prng.seed()

  A_pad_torch = torch.tensor(Apad)

  Aprime = randomized_hadamard_transform(A_pad_torch, prng.manual_seed(seed)).numpy()
  print(Aprime.shape)
  """

  total_budget = 0

  for seed in range(N_EXPERIMENTS):
    np.random.seed(seed)

    A = np.random.normal(loc=0, scale=0.1, size=(n, d))
    A_norm = np.linalg.norm(A, axis=1)

    for i in range(n):
      A[i] = A[i] / (A_norm[i] / 2.4)

    dPad = int(2**np.ceil(np.log2(d)))
    print(f'padded dimension: {dPad}')
    Apad = np.pad(A,((0,0),(0,dPad-d)),'constant',constant_values=0)

    prng = torch.Generator(device='cpu')
    hadamard_seed = prng.seed()

    A_pad_torch = torch.tensor(Apad)

    Aprime = randomized_hadamard_transform(A_pad_torch, prng.manual_seed(seed)).numpy()
    print(Aprime.shape)

    #generate datapoint
    best_index = int(np.random.uniform(0, 9.9))
    x = A[best_index] + np.random.normal(loc=0, scale=1e-3, size=d)

    xpad = np.pad(x,(0,dPad-d),'constant',constant_values=0)
    xprime = randomized_hadamard_transform(torch.tensor(xpad.T), prng.manual_seed(seed)).numpy().T

    print(xprime.shape)

    print("is product same?", np.allclose(A@x,Aprime@xprime))

    #calculate ground truth
    mu = A @ x
    mu -= np.max(mu)
    z = np.exp(mu) / np.sum(np.exp(mu))

    gain = n * np.sum(np.exp(2 * (mu - np.max(mu)))) / (np.sum(np.exp(mu - np.max(mu)))**2)
    print("gain:", gain)
    """
    transform_mu = 
    transform_gain = 
    """

    best_index_hat, z_hat, bandit_budget = ada_softmax_nb(A, x, beta, epsilon, delta, dPad, k)

    print("best_index:", best_index_hat)


    total_budget += bandit_budget

    cur_epsilon = np.abs(z_hat[best_index_hat] - z[best_index_hat]) / z[best_index_hat]
    print(z_hat[best_index_hat], z[best_index_hat], np.max(z))

    if cur_epsilon[0] > 1e-2:
      print(cur_epsilon)

    if cur_epsilon[0] <= epsilon and best_index_hat[0] == np.argmax(z): #ASSUMING K=1
      error_sum += cur_epsilon[0]
    elif best_index_hat[0] == np.argmax(z):
      wrong_approx_num += 1
      error_sum += cur_epsilon[0]
    else:
      #error_sum += cur_epsilon[0]
      wrong_approx_num += 1
      print(bandit_budget)
      print(z)
      print(z_hat)
      print(best_index_hat[0], np.argmax(z), cur_epsilon[0])


  imp_delta = wrong_approx_num / N_EXPERIMENTS
  average_budget = total_budget / N_EXPERIMENTS
  imp_epsilon = error_sum / N_EXPERIMENTS
  #gain_mean = gain_sum / N_EXPERIMENTS

  print("=>delta:", imp_delta)
  print("=>average budget:", average_budget)
  print("=>average error:", imp_epsilon)
  #print("=>wrong_approx_num:", wrong_approx_num)

  budget_list.append(average_budget)


plt.plot(dimension_list, budget_list, "r--.", label="adaptive_softmax")
plt.plot(dimension_list, n * np.array(dimension_list), "b--.", label="naive")
plt.legend()
plt.xlabel("dimension(n_features)")
plt.ylabel("number of samples taken")
plt.savefig("sample_complexity_plot.svg", bbox_inches="tight")
plt.yscale("log")
plt.plot(dimension_list, budget_list, "r--.", label="adaptive_softmax")
plt.plot(dimension_list, n * np.array(dimension_list), "b--.", label="naive")
plt.savefig("sample_complexity_log_plot.svg", bbox_inches="tight")