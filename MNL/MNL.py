# installing dependencies
import torch
torch.manual_seed(0)

import numpy as np
from time import time
import matplotlib.pyplot as plt
from numba import njit
from torchvision import datasets, transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

N_CLASSES = 10
#N_FEATURES = int(1e+6)
N_FEATURES = 0
N_DATA = 100
TEMP = 1
element_mu = 0
element_sigma = 1e-4
signal_strength = 5e+2


def get_data(n_repetition):
    EuroSAT_Train = datasets.EuroSAT(root="./data/",
                                 download=True,
                                 transform=transforms.ToTensor())

    data = list()
    labels = list()

    for i in range(27000):
        datum, label = EuroSAT_Train[i]

        flattened_datum_l = list()

        for color_channel in datum:
          for row in color_channel:
            flattened_datum_l.append(row)

        flattened_datum_l = n_repetition * flattened_datum_l

        flattened_datum = torch.nn.functional.normalize(torch.cat(flattened_datum_l).float(), dim=0)
        flattened_datum = (1 / n_repetition) * flattened_datum


        data.append(flattened_datum)
        labels.append(label)

    data = torch.stack(data).float()
    labels = torch.Tensor(labels).long()
    return data, labels


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
def compute_mip_batch_topk_ver2_warm_nb(atoms, query, sigma, delta, batch_size=16, k=1, mu=None, budget_vec=None):
    """
    does same thing as previous, but instead of doing multiplication between single element of A and x,
    it sequentially slices 'batch_size' elements from left to right, and performs inner product to
    pull an arm.
    """
    # TODO: Fix bruteforce failsafe(Refer to GPT implementation)

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

    C = np.divide(sigma * np.sqrt(2 * np.log(4 * n_atoms * d_used ** 2 / delta)), d_used + 1)

    solution_mask_before = solution_mask

    brute_force_threshold = np.ceil(atoms.shape[0] * 0.05)

    while (len(solutions) > brute_force_threshold and found_indices_num < k and np.max(
            d_used) < dim - batch_size):  # TODO: computing max everytime may degrade performance

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
            mu_exact[i] = atoms[atom_index, d_used[atom_index]:] @ query[d_used[atom_index]:]

        d_used[solutions] = dim

        mu = np.divide(mu_exact, d_used)

        # TODO: is there a way to avoid copy?
        mu_exact_search = mu_exact.copy()

        while found_indices_num < k:
            best_index = np.argmax(mu_exact_search)
            best_ind[found_indices_num] = best_index
            found_indices_num += 1
            mu_exact_search[best_index] = -np.inf

        n_samples += np.sum(dim - d_used[solution_mask])

    return best_ind, n_samples, mu, d_used

@njit
def estimate_softmax_normalization_warm_nb(atoms, query, beta, epsilon, delta, sigma, bruteforce=False):
    #TODO: when T0=d, return bruteforce

    n = atoms.shape[0]
    d = query.shape[0]
    used_samples = 0

    T0 = int(min(np.ceil(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)), d))

    #print("T0:", T0)

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

    mu_hat_exp_gamma = np.exp(mu_hat_aux / 2)
    gamma = mu_hat_exp_gamma / np.sum(mu_hat_exp_gamma)

    #import ipdb; ipdb.set_trace()
    T1 = 17 * np.log((6 * n) / delta) * n
    T2 = (32 * np.log((6 * n) / delta) * n * np.sum(mu_hat_exp_gamma)**2) / (epsilon * np.sum(mu_hat_exp_alpha))
    T3 = (16 * np.log(12 / delta)) / (epsilon ** 2)

    T = beta**2 * sigma**2 * (T1 + T2 + T3)

    #Experimental changes
    #normalized_sampling_distribution = (alpha + gamma) / np.sum(alpha + gamma)

    n_samples = np.ceil(np.minimum((alpha + gamma) * T + T0, d)).astype(np.int64)

    mu_hat_refined_aux = np.empty(n)

    for i in range(n):
        mu_hat_refined_aux[i] = atoms[i, T0:T0 + n_samples[i]] @ query[T0:T0 + n_samples[i]]

    mu_hat_refined = np.divide(mu_hat * T0 + mu_hat_refined_aux * d, np.maximum(n_samples, 1)) * beta

    #mu_hat_refined -= np.max(mu_hat_refined)

    mu_hat_refined_exp = np.exp(mu_hat_refined)
    S_hat = np.sum(mu_hat_refined_exp)

    return S_hat, mu_hat_refined, n_samples

    #Potential hazard: max value of mu_hat for alpha construction and max value of mu hat for S_hat estimation is different, which may result in incorrect sampling or other problems?

# adaSoftmax with warm start
@njit
def ada_softmax_nb(A, x, beta, epsilon, delta, n_sigma_sample, k):

    #TODO: repleace this with empirical bernstein bound, extend "warm start" to the sigma approximation layer
    sigma = approx_sigma_bound_nb(A, x, n_sigma_sample)

    print(sigma)

    S_hat, mu_hat, budget_vec = estimate_softmax_normalization_warm_nb(A, x, beta, epsilon / 2, delta / 3, sigma)

    #print("S estimate:", S_hat)

    #print("denominator budget:", np.sum(budget_vec))

    best_index_hat, budget_mip, mu_hat, budget_vec = compute_mip_batch_topk_ver2_warm_nb(A, x, sigma, delta / 3,
                                                                                      batch_size=16, k=k, mu=mu_hat,
                                                                                      budget_vec=budget_vec)

    #print("denominator + best arm identification:", np.sum(budget_vec))

    best_index_hat = best_index_hat[:k]

    n_arm_pull = int(min(
        np.ceil((288 * sigma ** 2 * beta ** 2 * np.log(6 / delta)) / (epsilon ** 2)),
        x.shape[0]
    ))

    #mu_additional = np.empty(k)

    for arm_index in best_index_hat:
        mu_additional = A[arm_index, budget_vec[arm_index]: n_arm_pull] @ x[budget_vec[arm_index]: n_arm_pull]
        mu_hat[arm_index] = (mu_hat[arm_index] * budget_vec[arm_index] + x.shape[0] * mu_additional) / max(budget_vec[arm_index], n_arm_pull, 1)

    #mu_hat[best_index_hat] += np.divide(x.shape[0] * mu_additional, np.maximum(1, n_arm_pull - budget_vec[best_index_hat]))

    budget_vec[best_index_hat] = np.maximum(budget_vec[best_index_hat], n_arm_pull)

    #print("total_budget:", np.sum(budget_vec))

    # y_best_hat = np.exp(beta * (mu_best_hat), dtype=np.float64)
    y_best_hat = np.exp(beta * (mu_hat))
    budget = np.sum(budget_vec)
    z = y_best_hat / S_hat

    return best_index_hat, z, budget

class CustomSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): # this is the numerically unstable version
        x = x - torch.max(x) # this is to counter overflow, though this is still numerically unstable due to the underflow if max(x) is large(if min(abs(x_i - max(x))) > (about)100)
        e = torch.exp(x, )
        out = e / torch.sum(e, dim=-1, keepdim=True)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        out = ctx.saved_tensors[0]
        return  grad * out * (1 - out)


class SimpleModel(torch.nn.Module):
    def __init__(self, temperature, data_dim, n_class):
        # simple classification model --> linear layer + softmax
        super().__init__()
        self.temperature = temperature
        self.linear = torch.nn.Linear(data_dim, n_class, bias=False, dtype=torch.float)
        #self.softmax = CustomSoftmax.apply
        self.softmax = torch.nn.Softmax()

    def forward(self, x):   # TODO: add torch.no)grad()
        x = x * self.temperature
        x = self.linear(x)
        out = self.softmax(x)
        return out

    def matmul(self, x):
        with torch.no_grad():
            out = self.linear(x)
        return out

    def get_linear_weight(self):
        return self.linear.weight.detach()

    def set_linear_weight(self, weight):
        self.linear.weight = torch.nn.parameter.Parameter(weight)

"""
def generate_data(shape, mu=0.0, std=0.5, bounds=(-1, 1), spiky_num = 0, spike_constant = 1):
    data = torch.clamp(mu + torch.randn(shape, dtype=torch.double) * std, bounds[0], bounds[1])
    if spiky_num > 0:
      #TODO: implement proper spiky vector generation
      pass
    return data
"""


def train(data, labels, model, max_iter=100):
    print("training")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(max_iter):
        running_loss = 0.0

        for i, datapoint in enumerate(data):
            optimizer.zero_grad()

            # compute gradients
            output = model(datapoint)
            loss = criterion(output, labels[i])
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

NUM_EXPERIMENTS = 50
NUM_DIMENSIONS = 50


dimension_list = list()
naive_time_list = list()
adaptive_time_list = list()
budget_list = list()
sigma_list = list()

data, labels = get_data(1)

data = data[:100]
labels = labels[:100]
model = SimpleModel(TEMP, data.shape[1], N_CLASSES)
train(data, labels, model, max_iter=5)

A = model.get_linear_weight()
original_A = A.detach()

for dim_constant in range(1, NUM_DIMENSIONS + 1):
  print("dim constant:", dim_constant)
  naive_time_list_aux = list()
  adaptive_time_list_aux = list()
  budget_list_aux = list()

  wrong_approx_num = 0
  budget_sum = 0
  error_sum = 0
  time_dif_sum = 0
  naive_time_sum = 0
  adaptive_time_sum = 0

  new_data = torch.cat(tuple([data]*dim_constant), 1)
  model = SimpleModel(TEMP, new_data.shape[1], N_CLASSES)
  train(new_data, labels, model, max_iter=30)
  A = model.get_linear_weight()
  A_ndarray = A.detach().numpy()


  epsilon = 0.1
  delta = 0.01
  top_k = 1

  for seed in range(NUM_EXPERIMENTS):
    x = new_data[seed % N_DATA]
    x_ndarray = x.detach().numpy()

    mu = A_ndarray @ x_ndarray
    print("gain:", N_CLASSES * np.sum(np.exp(2 * mu)) / (np.sum(np.exp(mu))**2))

    # naive softmax
    naive_start_time = time()
    z = model(x)
    naive_time = time() - naive_start_time
    naive_time_sum += naive_time

    #TEST
    naive_time_list_aux.append(naive_time)

    # AdaSoftmax

    adaptive_start_time = time()
    bandit_topk_indices, z_hat, bandit_budget = ada_softmax_nb(A_ndarray, x_ndarray, TEMP, epsilon, delta, dim_constant * data.shape[1], top_k)
    adaptive_time = time() - adaptive_start_time
    adaptive_time_sum += adaptive_time

    adaptive_time_list_aux.append(adaptive_time)

    numpy_z = z.detach().numpy()[labels]

    cur_epsilon = np.abs(z_hat[bandit_topk_indices] - np.max(numpy_z)) / np.max(numpy_z)

    if cur_epsilon[0] <= epsilon and bandit_topk_indices[0] == labels[seed % N_DATA]: #ASSUMING K=1
      error_sum += cur_epsilon[0]
    else:
      wrong_approx_num += 1

    budget_list_aux.append(bandit_budget)

  imp_delta = wrong_approx_num / NUM_EXPERIMENTS
  average_budget = budget_sum / NUM_EXPERIMENTS
  imp_epsilon = error_sum / NUM_EXPERIMENTS

  #naive_time_mean = np.mean(np.sort(np.array(naive_time_list_aux))[int(0.05 * NUM_EXPERIMENTS) : int(0.95 * NUM_EXPERIMENTS)])
  #adaptive_time_mean = np.mean(np.sort(np.array(adaptive_time_list_aux))[int(0.05 * NUM_EXPERIMENTS) : int(0.95 * NUM_EXPERIMENTS)])
  #budget_mean = np.mean(np.sort(np.array(budget_list_aux))[int(0.05 * NUM_EXPERIMENTS) : int(0.95 * NUM_EXPERIMENTS)])
  budget_mean = np.mean(budget_list_aux)


  print("=>delta:", imp_delta)
  print("=>average budget:", budget_mean)
  print("=>average error:", imp_epsilon)

  print("=>wrong_approx_num:", wrong_approx_num)

  dimension_list.append(N_FEATURES)
  #naive_time_list.append(naive_time_mean)
  #adaptive_time_list.append(adaptive_time_mean)
  budget_list.append(budget_mean)

plt.plot(12288 * np.arange(1, NUM_DIMENSIONS + 1), budget_list, "r--.", label="adaptive_softmax")
plt.plot(12288 * np.arange(1, NUM_DIMENSIONS + 1), N_CLASSES * 12288 * np.arange(1, NUM_DIMENSIONS + 1), "b--.", label="naive")
plt.legend()
plt.xlabel("dimension(n_features)")
plt.ylabel("number of samples taken")
plt.savefig("sample_complexity_plot.png")