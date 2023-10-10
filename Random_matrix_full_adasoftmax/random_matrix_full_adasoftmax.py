import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import torch
from hadamard_transform import randomized_hadamard_transform, hadamard_transform
np.random.seed(777)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptive_softmax'))
from adasoftmax import ada_softmax_nb


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