N_CLASSES = int(1e+1)
N_FEATURES = int(3e+5)
TEMP = 1
NUM_EXPERIMENTS = 100

import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, njit
import latex

plt.rcParams["text.usetex"] = True

@njit
def approx_sigma_bound_nb(A, x, n_sigma_sample):
    #NOTE: We're returning sigma as a sub-gaussian parameter(std of arm pull, not aij*xj)
    n_arms = A.shape[0]
    A_subset, x_subset = A[:, :n_sigma_sample], x[:n_sigma_sample]

    elmul = np.multiply(A[:, :n_sigma_sample], x[:n_sigma_sample])

    sigma = np.empty(n_arms)
    for i in range(n_arms):
        sigma[i] = np.std(elmul[i])

    return x.shape[0] * np.max(sigma)

@njit
def estimate_softmax_normalization_uniform(A, x, beta, epsilon, ground_truth, mu_max):
    #Maybe we can take true mu's max value to make this numerically stable properly?
    n_arms = A.shape[0]
    dimension = x.shape[0]
    target_error = epsilon * ground_truth
    n_strikes = 0

    total_budget = n_arms
    budget_per_arm = 1

    mu_hat = dimension * (A[:, :budget_per_arm] @ x[:budget_per_arm])
    estimate = np.sum(np.exp(beta * (mu_hat - mu_max)))
    if np.abs(estimate - ground_truth) <= target_error:
      n_strikes = 1

    cur_sampled_index = 1

    while n_strikes < 10:
      budget_per_arm *= 1.2
      if dimension - cur_sampled_index < budget_per_arm:
        budget_per_arm = dimension - cur_sampled_index



      new_sample = dimension * (A[:, cur_sampled_index : cur_sampled_index + budget_per_arm] @ x[cur_sampled_index : cur_sampled_index + budget_per_arm])
      mu_hat = (cur_sampled_index * mu_hat + new_sample) / (cur_sampled_index + budget_per_arm)
      estimate = np.sum(np.exp(beta * (mu_hat - mu_max)))
      cur_sampled_index += budget_per_arm
      total_budget += n_arms * budget_per_arm



      if np.abs(estimate - ground_truth) <= target_error:
        n_strikes += 1
      else:
        n_strikes = 0

    return total_budget

@njit
def estimate_softmax_normalization_adaptive_lv1(A, x, beta, epsilon, ground_truth, true_alpha, true_gamma, mu_max):
    n_arms = A.shape[0]
    dimension = x.shape[0]
    target_error = epsilon * ground_truth
    n_strikes = 0
    mu_hat = np.zeros(n_arms)

    cur_budget = n_arms
    budget_vec = (n_arms * true_alpha).astype(np.int64)

    for i in range(n_arms):
        mu_hat[i] = (dimension/np.maximum(1, budget_vec[i])) * (A[i, :budget_vec[i]] @ x[:budget_vec[i]])
    estimate = np.sum(np.exp(beta * (mu_hat - mu_max)))
    if np.abs(estimate - ground_truth) <= target_error:
      n_strikes = 1



    while n_strikes < 10:
        cur_budget *= 1.2
        new_budget_vec = (cur_budget * true_alpha).astype(np.int64)


        for i in range(n_arms):
            new_sample = dimension * (A[i, budget_vec[i] : budget_vec[i] + new_budget_vec[i]] @ x[budget_vec[i] : budget_vec[i] + new_budget_vec[i]])
            true_used_budget = np.minimum(new_budget_vec[i], dimension - budget_vec[i])
            mu_hat[i] = (mu_hat[i] * budget_vec[i] + new_sample) / (np.maximum(1, budget_vec[i] + true_used_budget))
            budget_vec[i] += true_used_budget

        estimate = np.sum(np.exp(beta * (mu_hat - mu_max)))

        if np.abs(estimate - ground_truth) <= target_error:
          n_strikes += 1
        else:
          n_strikes = 0

    return np.sum(budget_vec)

@njit
def estimate_softmax_normalization_adaptive_lv2(A, x, beta, epsilon, delta, sigma, ground_truth, mu_max):
    n = A.shape[0]
    d = x.shape[0]
    target_error = epsilon * ground_truth
    n_strikes = 0

    T0 = max(int(min(np.ceil(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)), d)), 1)
    budget_vec = np.full(n, T0)
    budget = T0

    mu_hat = (d / T0) * A[:, : T0] @ x[: T0]
    CI = np.sqrt(2 * sigma**2 * np.log(6*n/delta) / T0)

    alpha = np.exp(beta*(mu_hat - (CI + mu_max))) / np.sum(np.exp(beta*(mu_hat-(CI + mu_max))))
    gamma = np.exp((beta/2) * (mu_hat - (CI + mu_max))) / np.sum(np.exp((beta/2) * (mu_hat - (CI + mu_max))))

    estimate = np.sum(np.exp(beta * (mu_hat - mu_max)))


    if np.abs(estimate - ground_truth) <= target_error:
      n_strikes = 1


    while n_strikes < 10:
        budget *= 1.2
        CI = np.sqrt(2 * sigma**2 * np.log(6*n/delta) / budget_vec)
        alpha = np.exp(beta*(mu_hat - (CI + mu_max))) / np.sum(np.exp(beta*(mu_hat-(CI + mu_max))))
        gamma = np.exp((beta/2) * (mu_hat - (CI + mu_max))) / np.sum(np.exp((beta/2) * (mu_hat - (CI + mu_max))))

        new_budget_vec = (alpha * budget).astype(np.int64)

        for i in range(n):
            new_sample = d * (A[i, budget_vec[i] : budget_vec[i] + new_budget_vec[i]] @ x[budget_vec[i] : budget_vec[i] + new_budget_vec[i]])

            true_used_budget = np.minimum(new_budget_vec[i], d - budget_vec[i])

            mu_hat[i] = (mu_hat[i] * budget_vec[i] + new_sample) / (np.maximum(1, budget_vec[i] + true_used_budget))
            budget_vec[i] += true_used_budget

        estimate = np.sum(np.exp((mu_hat - mu_max)))

        if np.abs(estimate - ground_truth) <= target_error:
          n_strikes += 1
        else:
          n_strikes = 0

    error = np.abs(estimate - ground_truth) / ground_truth

    if error > epsilon:
      print("error over bound")
      print("error_diff:", error - epsilon)

    return np.sum(budget_vec)

c1 = np.linspace(1, 1.1, num=6, endpoint=False)
c2 = np.linspace(1.1, 1.35, num=40, endpoint=False)
c3 = np.linspace(1.35, 1.5, num=4)
c_list = np.concatenate((c1, c2, c3))

# for plotting
ideal_gain_list = list()
lv1_gain_list = list()
lv2_gain_list = list()
lv3_gain_list = list()

for i, c in enumerate(c_list):
  print(i)

  np.random.seed(42)

  true_mu = np.ones(N_CLASSES)
  true_mu[1] = true_mu[1] * c
  true_mu = true_mu / (c/20)    # for constant sigma

  theoretical_gain = N_CLASSES * np.sum(np.exp(2 * true_mu)) / (np.sum(np.exp(true_mu))**2)
  ideal_gain_list.append(theoretical_gain)

  print("--------------------------------")
  print("theory:", theoretical_gain)

  total_uniform_budget = 0
  total_level1_budget = 0
  total_level2_budget = 0
  total_level3_budget = 0


  x = np.random.uniform(low=0.94, high=1, size = N_FEATURES)
  Z = np.random.normal(loc=0, scale=1/N_FEATURES, size=(N_CLASSES, N_FEATURES))
  A = np.outer(true_mu, x)/np.sum(x**2) + Z
  A = A - np.outer(A@x - true_mu, np.ones(N_FEATURES)/np.sum(x))

  for i in range(NUM_EXPERIMENTS):

    np.random.seed(20+i)

    perm = np.random.permutation(N_FEATURES)
    A = A[:, perm]
    x = x[perm]

    epsilon = 5e-3
    delta = 0.01
    k=1
    n_sigma_sample = N_FEATURES

    mu_max = 0

    ground_truth = np.sum(np.exp(TEMP * (true_mu - mu_max)))

    true_sigma = approx_sigma_bound_nb(A, x, N_FEATURES)

    mu_hat_lower_bound = TEMP * (true_mu - mu_max)

    true_alpha = np.exp(mu_hat_lower_bound) / np.sum(np.exp(mu_hat_lower_bound))

    true_gamma = np.exp(0.5 * mu_hat_lower_bound) / np.sum(np.exp(0.5 * mu_hat_lower_bound))



    uniform_budget = estimate_softmax_normalization_uniform(A, x, TEMP, epsilon, ground_truth, mu_max)
    total_uniform_budget += uniform_budget

    level1_budget = estimate_softmax_normalization_adaptive_lv1(A, x, TEMP, epsilon, ground_truth, true_alpha, true_gamma, mu_max)
    total_level1_budget += level1_budget

    level2_budget = estimate_softmax_normalization_adaptive_lv2(A, x, TEMP, epsilon, delta, true_sigma, ground_truth, mu_max)
    total_level2_budget += level2_budget


  uniform_budget_mean = total_uniform_budget / NUM_EXPERIMENTS
  level1_budget_mean = total_level1_budget / NUM_EXPERIMENTS
  level2_budget_mean = total_level2_budget / NUM_EXPERIMENTS

  print("uniform budget:", uniform_budget_mean)
  print("level1 budget:", level1_budget_mean)
  print("level2 budget:", level2_budget_mean)

  print("theoretical gain:", theoretical_gain)
  print("lvl1 gain:", uniform_budget_mean / level1_budget_mean)
  print("lvl2 gain:", uniform_budget_mean / level2_budget_mean)

  lv1_gain_list.append(uniform_budget_mean / level1_budget_mean)
  lv2_gain_list.append(uniform_budget_mean / level2_budget_mean)

plt.plot(ideal_gain_list, ideal_gain_list, "b--", label="theoretical_baseline")
plt.scatter(ideal_gain_list, lv1_gain_list, color="green", label=r"adaptive gain with true $\alpha$ and stopping condition")
plt.scatter(ideal_gain_list, lv2_gain_list, color="yellow", label="adaptive gain with true stopping condition")
plt.xlabel("theoretical gain")
plt.ylabel("empirical_gain")
plt.legend()

plt.savefig("random_matrix_uniform_versus_adaptive.svg")