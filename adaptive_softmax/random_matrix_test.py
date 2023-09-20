import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def approx_sigma_bound_nb(A, x, n_sigma_sample):
    #NOTE: We're returning sigma as a sub-gaussian parameter(std of arm pull, not aij*xj)
    n_arms = A.shape[0]
    A_subset, x_subset = A[:, :n_sigma_sample], x[:n_sigma_sample]

    elmul = np.multiply(A[:, :n_sigma_sample], x[:n_sigma_sample])
    #sigma = np.sqrt(np.mean(np.square(elmul.T - np.mean(elmul, axis=1)), axis=0))
    arm_pull_mean = (A_subset @ x_subset) / n_sigma_sample

    sigma = np.empty(n_arms)
    for i in range(n_arms):
        sigma[i] = np.sqrt(np.mean(np.square(elmul[i] - arm_pull_mean[i])))

    return x.shape[0] * np.max(sigma)

@njit
def estimate_softmax_normalization_uniform(A, x, beta, epsilon, ground_truth, mu_max):
    #Maybe we can take true mu's max value to make this numerically stable properly?
    n_arms = A.shape[0]
    dimension = x.shape[0]
    target_error = epsilon * ground_truth

    budget_per_arm = 1
    cur_sampled_index = 1

    total_budget = n_arms

    mu_hat = dimension * (A[:, :budget_per_arm] @ x[:budget_per_arm])

    #mu_hat -= np.max(mu_hat)

    estimate = np.sum(np.exp(beta * (mu_hat - mu_max)))

    while np.abs(estimate - ground_truth) > target_error:
      #print("bpa:", budget_per_arm * n_arms)
      budget_per_arm *= 1.5
      #budget_per_arm += 1

      if dimension - cur_sampled_index < budget_per_arm:
        budget_per_arm = dimension - cur_sampled_index

      new_sample = dimension * (A[:, cur_sampled_index : cur_sampled_index + budget_per_arm] @ x[cur_sampled_index : cur_sampled_index + budget_per_arm])
      #mu_hat -= np.max(mu_hat)
      mu_hat = (cur_sampled_index * mu_hat + new_sample) / (cur_sampled_index + budget_per_arm)
      estimate = np.sum(np.exp(beta * (mu_hat - mu_max)))

      cur_sampled_index += budget_per_arm
      total_budget += n_arms * budget_per_arm

      #print(cur_sampled_index)

    #print(np.abs(estimate - ground_truth) / ground_truth)

    return total_budget

@njit
def estimate_softmax_normalization_adaptive_lv1(A, x, beta, epsilon, ground_truth, true_alpha, true_gamma, mu_max):
    n_arms = A.shape[0]
    dimension = x.shape[0]

    target_error = epsilon * ground_truth

    mu_hat = np.zeros(n_arms)

    cur_budget = n_arms

    budget_vec = (n_arms * true_alpha).astype(np.int64)

    for i in range(n_arms):
        mu_hat[i] = (dimension/np.maximum(1, budget_vec[i])) * (A[i, :budget_vec[i]] @ x[:budget_vec[i]])

    estimate = np.sum(np.exp(beta * (mu_hat - mu_max)))

    while np.abs(estimate - ground_truth) > target_error:

        cur_budget *= 1.5
        #print("lv1:", cur_budget)
        #cur_budget += 1

        new_budget_vec = (cur_budget * (true_alpha + true_gamma)).astype(np.int64)

        for i in range(n_arms):
            new_sample = dimension * (A[i, budget_vec[i] : budget_vec[i] + new_budget_vec[i]] @ x[budget_vec[i] : budget_vec[i] + new_budget_vec[i]])

            #print(new_sample)

            true_used_budget = np.minimum(new_budget_vec[i], dimension - budget_vec[i])

            #print((mu_hat[i] * budget_vec[i] + new_sample) / (np.maximum(1, budget_vec[i] + true_used_budget)))

            mu_hat[i] = (mu_hat[i] * budget_vec[i] + new_sample) / (np.maximum(1, budget_vec[i] + true_used_budget))
            budget_vec[i] += true_used_budget

        estimate = np.sum(np.exp(beta * (mu_hat - mu_max)))

    #print(np.abs(estimate - ground_truth) / ground_truth)
    #print(budget)

    return np.sum(budget_vec)

def estimate_softmax_normalization_adaptive_lv2(A, x, beta, epsilon, delta, sigma, ground_truth, mu_max):
    n = A.shape[0]
    d = x.shape[0]

    target_error = epsilon * ground_truth

    T0 = max(int(min(np.ceil(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)), d)), 1)
    #T0 = int(0.01*d)

    budget_vec = np.full(n, T0)

    mu_hat = (d / T0) * A[:, : T0] @ x[: T0]

    CI = np.sqrt(2 * sigma * np.log(6*n/delta) / T0)

    alpha = np.exp(beta*(mu_hat - (CI + mu_max))) / np.sum(np.exp(beta*(mu_hat-(CI + mu_max))))

    gamma = np.exp((beta/2) * (mu_hat - (CI + mu_max))) / np.sum(np.exp((beta/2) * (mu_hat - (CI + mu_max))))

    budget = T0 #Assuming that T0 is "sufficiently" small here

    estimate = np.sum(np.exp(beta * (mu_hat - mu_max)))

    while np.abs(estimate - ground_truth) > target_error:
        #print(budget)
        budget *= 1.5
        #budget += 1

        new_budget_vec = ((alpha + gamma) * budget).astype(np.int64)

        for i in range(n):
            new_sample = d * (A[i, budget_vec[i] : budget_vec[i] + new_budget_vec[i]] @ x[budget_vec[i] : budget_vec[i] + new_budget_vec[i]])

            #print(new_sample)

            true_used_budget = np.minimum(new_budget_vec[i], d - budget_vec[i])

            mu_hat[i] = (mu_hat[i] * budget_vec[i] + new_sample) / (np.maximum(1, budget_vec[i] + true_used_budget))
            budget_vec[i] += true_used_budget

        estimate = np.sum(np.exp((mu_hat - mu_max)))

    print("lv2 error:", np.abs(estimate - ground_truth) / ground_truth)

    return np.sum(budget_vec)

@njit
def estimate_softmax_normalization_adaptive_lv3(atoms, query, beta, epsilon, delta, sigma, mu_max, ground_truth):
    #Ground truth should NOT be given to level 3, so ground truth is not used during estimation. Ground truth is fed to this function just to evaluate epsilon

    n = atoms.shape[0]
    d = query.shape[0]
    used_samples = 0

    T0 = max(int(min(np.ceil(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)), d)), 1)
    #T0 = int(0.01*d)

    mu_hat = (d / T0) * (atoms[:, :T0] @ query[:T0])
    C = (2 * sigma ** 2 * np.log(6 * n / delta) / T0) ** 0.5

    n_samples = np.zeros(n, dtype=np.int64) + T0

    #Maybe this is better?
    #mu_hat -= np.min(mu_hat)

    mu_hat_aux = (mu_hat - C) * beta
    #mu_hat_aux -= np.max(mu_hat_aux)
    #mu_hat_aux = (mu_hat - C) * beta
    mu_hat_exp_alpha = np.exp(mu_hat_aux)
    alpha = mu_hat_exp_alpha / np.sum(mu_hat_exp_alpha)

    #print("alpha:", alpha)

    mu_hat_exp_gamma = np.exp(mu_hat_aux / 2)
    gamma = mu_hat_exp_gamma / np.sum(mu_hat_exp_gamma)

    #print("T0:", T0)
    #print("alpha estimate:", alpha)

    #import ipdb; ipdb.set_trace()

    T = 0.1 * beta**2 * sigma**2 * (
            17 * np.log((6 * n) / delta) * n
            + (32 * np.log((6 * n) / delta) * n * np.sum(mu_hat_exp_gamma)**2) / (epsilon * np.sum(mu_hat_exp_alpha))
            + (16 * np.log(12 / delta)) / epsilon ** 2
    )

    #print(17 * np.log((6 * n) / delta) * n, (32 * np.log((6 * n) / delta) * n * np.sum(mu_hat_exp_gamma)**2) / (epsilon * np.sum(mu_hat_exp_alpha)), (16 * np.log(12 / delta)) / epsilon ** 2, np.sum(mu_hat_exp_gamma)**2, np.sum(mu_hat_exp_alpha))
    #print("T:", T)

    n_samples = np.ceil(np.minimum((alpha + gamma) * T + T0, d)).astype(np.int64)

    #print("n_samples:", n_samples)

    mu_hat_refined_aux = np.empty(n)

    for i in range(n):
        mu_hat_refined_aux[i] = atoms[i, T0:T0 + n_samples[i]] @ query[T0:T0 + n_samples[i]]

    mu_hat_refined = np.divide(mu_hat * T0 + mu_hat_refined_aux * d, np.maximum(n_samples, 1)) * beta

    mu_hat_refined_exp = np.exp(mu_hat_refined - beta * mu_max)
    S_hat = np.sum(mu_hat_refined_exp)

    #print(mu_hat_refined_exp)

    #print("S_hat:", S_hat)
    print("lv3 error:", np.abs(S_hat - ground_truth) / ground_truth)

    return np.sum(n_samples)


total_uniform_budget = 0
total_level1_budget = 0
total_level2_budget = 0
total_level3_budget = 0

c_list = list()
theory_gain_list = list()
lv1_gain_list = list()
lv2_gain_list = list()
lv3_gain_list = list()

#for plot that Tavor suggested(probably?)
empirical_ideal_gain_list = list()
lv1_complexity_list = list()
lv2_complexity_list = list()
lv3_complexity_list = list()


theory_gain_list_aux = list()
lv1_gain_list_aux = list()
lv2_gain_list_aux = list()
lv3_gain_list_aux = list()


N_CLASSES = int(5e+1)
N_FEATURES = int(1e+6)
TEMP = 1
NUM_EXPERIMENTS = 20

for i in range(NUM_EXPERIMENTS):
  print(i)
  #np.random.seed(i)
  #A = np.random.normal(loc=0, scale=1 / N_FEATURES * c, size=(N_CLASSES, N_FEATURES))

  c = np.random.uniform(low=0.0, high=150)
  print("c:", c)

  
  true_mu = np.ones(N_CLASSES)
  true_mu[1] = true_mu[1] * c

  true_mu = true_mu / (c/5)
  #true_mu = np.random.pareto(2, size=(N_CLASSES))

  """
  true_mu = np.zeros(N_CLASSES)
  true_mu[1] = c
  """

  x = np.random.uniform(low=0.5, high=1, size = N_FEATURES)

  Z = np.random.normal(loc=0, scale=1/N_FEATURES, size=(N_CLASSES, N_FEATURES))

  A = np.outer(true_mu, x)/np.sum(x**2) + Z

  A = A - np.outer(A@x - true_mu, np.ones(N_FEATURES)/np.sum(x))

  #print("matrix generated")

  epsilon = 5e-3
  delta = 0.01
  k=1
  n_sigma_sample = N_FEATURES

  #x = c*A[1] + np.random.normal(loc=0, scale=1e-2 / N_FEATURES * c)
  #print(x.shape)

  mu = A @ x

  mu_max = 0

  #print(mu)

  #mu_max = np.max(mu)

  ground_truth = np.sum(np.exp(TEMP * (mu - mu_max)))

  theoretical_gain = N_CLASSES * np.sum(np.exp(2 * (mu - mu_max))) / (np.sum(np.exp(mu - mu_max)))**2
  theory_gain_list_aux.append(theoretical_gain)

  print("theory:", theoretical_gain)

  true_sigma = approx_sigma_bound_nb(A, x, N_FEATURES)
  print("sigma:", true_sigma)
  #true_sigma = 1
  CI = np.sqrt(2 * true_sigma * np.log(6*N_CLASSES/delta) / N_FEATURES)

  mu_hat_lower_bound = TEMP * (mu - CI - mu_max)

  true_alpha = np.exp(mu_hat_lower_bound) / np.sum(np.exp(mu_hat_lower_bound))

  true_gamma = np.exp(0.5 * mu_hat_lower_bound) / np.sum(np.exp(0.5 * mu_hat_lower_bound))

  print(true_alpha)
  #print(mu)

  uniform_budget = estimate_softmax_normalization_uniform(A, x, TEMP, epsilon / 2, ground_truth, mu_max)

  total_uniform_budget += uniform_budget

  #print("uniform")
  print("uniform_budget:", uniform_budget)

  level1_budget = estimate_softmax_normalization_adaptive_lv1(A, x, TEMP, epsilon / 2, ground_truth, true_alpha, true_gamma, mu_max)

  print("level1 budget:", level1_budget)

  total_level1_budget += level1_budget

  lv1_gain_list_aux.append(uniform_budget / level1_budget)

  print("level1 gain:", uniform_budget / level1_budget)

  level2_budget = estimate_softmax_normalization_adaptive_lv2(A, x, TEMP, epsilon / 2, delta, true_sigma, ground_truth, mu_max)

  lv2_gain_list_aux.append(uniform_budget / level2_budget)

  total_level2_budget += level2_budget

  print("level2 gain:", uniform_budget / level2_budget)

  level3_budget = estimate_softmax_normalization_adaptive_lv3(A, x, TEMP, epsilon, delta, true_sigma, mu_max, ground_truth)

  lv3_gain_list_aux.append(uniform_budget / level3_budget)
  print("lv3 gain:", uniform_budget / level3_budget)

  total_level3_budget += level3_budget

  empirical_ideal_gain_list.append(uniform_budget / level1_budget)
  lv1_complexity_list.append((theoretical_gain, uniform_budget / level1_budget))
  lv2_complexity_list.append((theoretical_gain, uniform_budget / level2_budget))
  lv3_complexity_list.append((theoretical_gain, uniform_budget / level3_budget))


uniform_budget_mean = total_uniform_budget / NUM_EXPERIMENTS
level1_budget_mean = total_level1_budget / NUM_EXPERIMENTS
level2_budget_mean = total_level2_budget / NUM_EXPERIMENTS
level3_budget_mean = total_level3_budget / NUM_EXPERIMENTS

c_list.append(c)
theory_gain_list.append(np.median(np.array(theory_gain_list_aux)))
lv1_gain_list.append(np.median(np.array(lv1_gain_list_aux)))
lv2_gain_list.append(np.median(np.array(lv2_gain_list_aux)))
lv3_gain_list.append(np.median(np.array(lv3_gain_list_aux)))


print("uniform:", uniform_budget_mean)
print("level1:", level1_budget_mean)
print("level2:", level2_budget_mean)
print("level3:", level3_budget_mean)
print("lvl1 gain:", uniform_budget_mean / level1_budget_mean)
#print("lvl2 gain:", uniform_budget_mean / level2_budget_mean)
print("lvl3 gain:", uniform_budget_mean / level3_budget_mean)