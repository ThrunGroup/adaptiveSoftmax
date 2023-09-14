import numpy as np
import matplotlib.pyplot as plt
from numba import njit

N_CLASSES = int(1e+1)
N_FEATURES = int(1e+7)
N_DATA = 100
TEMP = 1
NUM_EXPERIMENTS = 100
element_mu = 0
element_sigma = 1e-4

@njit
def estimate_softmax_normalization_uniform(A, x, beta, epsilon, ground_truth):
    #Maybe we can take true mu's max value to make this numerically stable properly?
    n_arms = A.shape[0]
    dimension = x.shape[0]
    target_error = epsilon * ground_truth

    budget_per_arm = 1
    cur_sampled_index = 1

    total_budget = n_arms

    mu_hat = (dimension*beta) * (A[:, :budget_per_arm] @ x[:budget_per_arm])

    #mu_hat -= np.max(mu_hat)

    estimate = np.sum(np.exp(mu_hat))

    while np.abs(estimate - ground_truth) > target_error:
      budget_per_arm *= 2

      if dimension - cur_sampled_index < budget_per_arm:
        budget_per_arm = dimension - cur_sampled_index

      new_sample = (dimension*beta) * (A[:, cur_sampled_index : cur_sampled_index + budget_per_arm] @ x[cur_sampled_index : cur_sampled_index + budget_per_arm])
      #mu_hat -= np.max(mu_hat)
      mu_hat = (cur_sampled_index * mu_hat + new_sample) / (cur_sampled_index + budget_per_arm)
      estimate = np.sum(np.exp(mu_hat))

      cur_sampled_index += budget_per_arm
      total_budget += n_arms * budget_per_arm

      #print(cur_sampled_index)

    #print(np.abs(estimate - ground_truth) / ground_truth)

    return total_budget

@njit
def estimate_softmax_normalization_adaptive_lv1(A, x, beta, epsilon, ground_truth, true_alpha):
    n_arms = A.shape[0]
    dimension = x.shape[0]

    target_error = epsilon * ground_truth

    mu_hat = np.zeros(n_arms)

    cur_budget = n_arms

    budget_vec = (n_arms * true_alpha).astype(np.int64)

    for i in range(n_arms):
        mu_hat[i] = (dimension/np.maximum(1, budget_vec[i])) * (A[i, :budget_vec[i]] @ x[:budget_vec[i]])

    estimate = np.sum(np.exp(mu_hat))

    while np.abs(estimate - ground_truth) > target_error:
        cur_budget *= 2

        new_budget_vec = (cur_budget * true_alpha).astype(np.int64)

        for i in range(n_arms):
            new_sample = dimension * (A[i, budget_vec[i] : budget_vec[i] + new_budget_vec[i]] @ x[budget_vec[i] : budget_vec[i] + new_budget_vec[i]])

            #print(new_sample)

            true_used_budget = np.minimum(new_budget_vec[i], dimension - budget_vec[i])

            #print((mu_hat[i] * budget_vec[i] + new_sample) / (np.maximum(1, budget_vec[i] + true_used_budget)))

            mu_hat[i] = (mu_hat[i] * budget_vec[i] + new_sample) / (np.maximum(1, budget_vec[i] + true_used_budget))
            budget_vec[i] += true_used_budget

        estimate = np.sum(np.exp(mu_hat))

    #print(np.abs(estimate - ground_truth) / ground_truth)
    #print(budget)

    return np.sum(budget_vec)

@njit
def estimate_softmax_normalization_adaptive_lv2(A, x, beta, epsilon, delta, sigma, ground_truth):
    n = A.shape[0]
    d = x.shape[0]

    target_error = epsilon * ground_truth

    T0 = int(min(np.ceil(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)), d))
    #T0 = int(0.001*d)
    print("T0:", T0)

    budget_vec = np.full(n, T0)

    mu_hat = (d / T0) * A[:, : T0] @ x[: T0]

    CI = np.sqrt(2 * sigma * np.log(6*n/delta) / T0)

    alpha = np.exp(beta*(mu - CI)) / np.sum(np.exp(beta*(mu_hat-CI)))

    budget = T0 #Assuming that T0 is "sufficiently" small here

    estimate = np.sum(np.exp(beta * mu_hat))

    while np.abs(estimate - ground_truth) > target_error:
        budget *= 2

        new_budget_vec = (alpha * budget).astype(np.int64)

        for i in range(n):
            new_sample = d * (A[i, budget_vec[i] : budget_vec[i] + new_budget_vec[i]] @ x[budget_vec[i] : budget_vec[i] + new_budget_vec[i]])

            #print(new_sample)

            true_used_budget = np.minimum(new_budget_vec[i], d - budget_vec[i])

            mu_hat[i] = (mu_hat[i] * budget_vec[i] + new_sample) / (np.maximum(1, budget_vec[i] + true_used_budget))
            budget_vec[i] += true_used_budget

        estimate = np.sum(np.exp(mu_hat))

    print(np.abs(estimate - ground_truth) / ground_truth)

    return np.sum(budget_vec)

@njit
def estimate_softmax_normalization_adaptive_lv2(A, x, beta, epsilon, delta, sigma, ground_truth):
    n = A.shape[0]
    d = x.shape[0]

    target_error = epsilon * ground_truth

    #T0 = int(min(np.ceil(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)), d))
    T0 = int(0.001*d)

    budget_vec = np.full(n, T0)

    mu_hat = (d / T0) * A[:, : T0] @ x[: T0]

    CI = np.sqrt(2 * sigma * np.log(6*n/delta) / T0)

    alpha = np.exp(beta*(mu - CI)) / np.sum(np.exp(beta*(mu_hat-CI)))

    budget = T0 #Assuming that T0 is "sufficiently" small here

    estimate = np.sum(np.exp(beta * mu_hat))

    while np.abs(estimate - ground_truth) > target_error:
        budget *= 2

        new_budget_vec = (alpha * budget).astype(np.int64)

        for i in range(n):
            new_sample = d * (A[i, budget_vec[i] : budget_vec[i] + new_budget_vec[i]] @ x[budget_vec[i] : budget_vec[i] + new_budget_vec[i]])

            #print(new_sample)

            true_used_budget = np.minimum(new_budget_vec[i], d - budget_vec[i])

            mu_hat[i] = (mu_hat[i] * budget_vec[i] + new_sample) / (np.maximum(1, budget_vec[i] + true_used_budget))
            budget_vec[i] += true_used_budget

        estimate = np.sum(np.exp(mu_hat))

    print(np.abs(estimate - ground_truth) / ground_truth)

    return np.sum(budget_vec)

def estimate_softmax_normalization_adaptive_lv3(atoms, query, beta, epsilon, delta, sigma):
    #TODO: when T0=d, return bruteforce

    n = atoms.shape[0]
    d = query.shape[0]
    used_samples = 0

    #T0 = int(min(np.ceil(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)), d))
    T0 = int(0.001*d)

    mu_hat = (d / T0) * (atoms[:, :T0] @ query[:T0])
    C = (2 * sigma ** 2 * np.log(6 * n / delta) / T0) ** 0.5

    n_samples = np.zeros(n, dtype=np.int64) + T0

    #Maybe this is better?
    #mu_hat -= np.min(mu_hat)

    mu_hat_aux = (mu_hat - C) * beta
    mu_hat_aux -= np.max(mu_hat_aux)
    #mu_hat_aux = (mu_hat - C) * beta
    mu_hat_exp_alpha = np.exp(mu_hat_aux)
    alpha = mu_hat_exp_alpha / np.sum(mu_hat_exp_alpha)

    #print("alpha:", alpha)

    mu_hat_exp_gamma = np.exp(mu_hat_aux / 2)
    gamma = mu_hat_exp_gamma / np.sum(mu_hat_exp_gamma)

    #import ipdb; ipdb.set_trace()

    T = beta**2 * sigma**2 * (
            17 * np.log((6 * n) / delta) * n
            + (32 * np.log((6 * n) / delta) * n * np.sum(mu_hat_exp_gamma)**2) / epsilon * np.sum(mu_hat_exp_alpha)
            + (16 * np.log(12 / delta)) / epsilon ** 2
    )

    n_samples = np.ceil(np.minimum((alpha + gamma) * T + T0, d)).astype(np.int64)

    mu_hat_refined_aux = np.empty(n)

    for i in range(n):
        mu_hat_refined_aux[i] = atoms[i, T0:T0 + n_samples[i]] @ query[T0:T0 + n_samples[i]]

    mu_hat_refined = np.divide(mu_hat * T0 + mu_hat_refined_aux * d, np.maximum(n_samples, 1)) * beta

    mu_hat_refined -= np.max(mu_hat_refined)

    mu_hat_refined_exp = np.exp(mu_hat_refined)
    S_hat = np.sum(mu_hat_refined_exp)

    return np.sum(n_samples)


total_uniform_budget = 0
total_level1_budget = 0
total_level2_budget = 0
total_level3_budget = 0

for i in range(100):
    print(i)
    A = np.random.normal(loc=0, scale=5e-3, size=(N_CLASSES, N_FEATURES))
    c = 0.1

    epsilon = 0.1
    delta = 0.01
    k=1
    n_sigma_sample = N_FEATURES

    x = c*A[1] + np.random.normal(loc=0, scale=1e-5)

    mu = A @ x

    #mu -= np.max(mu)
    ground_truth = np.sum(np.exp(mu))
    true_sigma = approx_sigma_bound_nb(A, x, N_FEATURES)
    CI = np.sqrt(2 * true_sigma * np.log(6*N_CLASSES/delta) / N_FEATURES)

    mu_hat_lower_bound = TEMP * (mu - CI)

    true_alpha = np.exp(mu_hat_lower_bound) / np.sum(np.exp(mu_hat_lower_bound))
    true_gamma = np.exp(0.5 * mu_hat_lower_bound) / np.sum(np.exp(0.5 * mu_hat_lower_bound))

    #print(true_alpha)
    #print(mu)

    uniform_budget = estimate_softmax_normalization_uniform(A, x, TEMP, epsilon / 2, ground_truth)

    total_uniform_budget += uniform_budget

    level1_budget = estimate_softmax_normalization_adaptive_lv1(A, x, TEMP, epsilon / 2, ground_truth, true_alpha)

    total_level1_budget += level1_budget

    level2_budget = estimate_softmax_normalization_adaptive_lv2(A, x, TEMP, epsilon / 2, delta, true_sigma, ground_truth)

    total_level2_budget += level2_budget

    level3_budget = estimate_softmax_normalization_adaptive_lv3(A, x, TEMP, epsilon, delta, true_sigma)
    print(level3_budget)

    total_level3_budget += level3_budget

uniform_budget_mean = total_uniform_budget / 100
level1_budget_mean = total_level1_budget / 100
level2_budget_mean = total_level2_budget / 100
level3_budget_mean = total_level3_budget / 100


print("uniform:", uniform_budget_mean)
print("level1:", level1_budget_mean)
print("level2:", level2_budget_mean)
print("level3:", level3_budget_mean)
print("lvl1 gain:", uniform_budget_mean / level1_budget_mean)
print("lvl2 gain:", uniform_budget_mean / level2_budget_mean)
print("lvl3 gain:", uniform_budget_mean / level3_budget_mean)