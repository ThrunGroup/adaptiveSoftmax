# installing dependencies
import torch
torch.manual_seed(0)

import numpy as np
from time import time
from adasoftmax import AdaSoftmax

N_CLASSES = 10
N_FEATURES = int(1e+6)
N_DATA = 100
TEMP = 1
NUM_EXPERIMENTS = 1000
element_mu = 0
element_sigma = 5e-1

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
    def __init__(self, temperature):
        # simple classification model --> linear layer + softmax
        super().__init__()
        self.temperature = temperature
        self.linear = torch.nn.Linear(N_FEATURES, N_CLASSES, bias=False, dtype=torch.double)
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


def generate_data(shape, mu=0.0, std=0.5, bounds=(-1, 1), spiky_num = 0, spike_constant = 1):
    data = torch.clamp(mu + torch.randn(shape, dtype=torch.double) * std, bounds[0], bounds[1])
    if spiky_num > 0:
      #TODO: implement proper spiky vector generation
      pass
    return data

def train(data, labels, model, max_iter=100):
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

dimension_list = list()
naive_time_list = list()
adaptive_time_list = list()
budget_list = list()

base_dim = int(1e+6)

algo = AdaSoftmax()

for dim_constant in range(0, 1):
  naive_time_list_aux = list()
  adaptive_time_list_aux = list()
  budget_list_aux = list()

  wrong_approx_num = 0
  budget_sum = 0
  error_sum = 0
  time_dif_sum = 0
  N_FEATURES = dim_constant * int(5e+5) + base_dim
  naive_time_sum = 0
  adaptive_time_sum = 0

  model = SimpleModel(TEMP)
  np.random.seed(dim_constant)
  data = generate_data((N_DATA, N_FEATURES), mu=element_mu, std=element_sigma, bounds=(element_mu - 5 * element_sigma, element_mu + 5 * element_sigma))
  #labels = torch.randint(N_CLASSES, size=(N_DATA,))

  labels = torch.ones(N_DATA).type(torch.long) * 4
  train(data, labels, model, max_iter=2)

  A = list(model.parameters())[0]
  A_ndarray = A.detach().numpy()

  x = data[0]
  x_ndarray = x.detach().numpy()

  epsilon = 0.1
  delta = 0.1
  top_k = 1

  for seed in range(NUM_EXPERIMENTS):
    if seed % 100 == 0:
      print(seed)
    x = data[seed % N_DATA]
    x_ndarray = x.detach().numpy()

    # naive softmax
    naive_start_time = time()
    z = model(x)
    naive_time = time() - naive_start_time
    naive_time_sum += naive_time

    #TEST
    naive_time_list_aux.append(naive_time)

    # AdaSoftmax

    adaptive_start_time = time()
    bandit_topk_indices, z_hat, bandit_budget = algo.ada_softmax(A_ndarray, x_ndarray, TEMP, epsilon, delta, N_FEATURES, top_k)
    adaptive_time = time() - adaptive_start_time
    adaptive_time_sum += adaptive_time

    adaptive_time_list_aux.append(adaptive_time)

    numpy_z = z.detach().numpy()[bandit_topk_indices]

    cur_epsilon = np.abs(z_hat[bandit_topk_indices] - np.max(numpy_z)) / np.max(numpy_z)

    error_sum += cur_epsilon[0]

    if cur_epsilon > epsilon:
      wrong_approx_num += 1

    budget_list_aux.append(bandit_budget)

  imp_delta = wrong_approx_num / NUM_EXPERIMENTS
  average_budget = budget_sum / NUM_EXPERIMENTS
  imp_epsilon = error_sum / NUM_EXPERIMENTS

  naive_time_mean = np.mean(np.sort(np.array(naive_time_list_aux))[int(0.05 * NUM_EXPERIMENTS) : int(0.95 * NUM_EXPERIMENTS)])
  adaptive_time_mean = np.mean(np.sort(np.array(adaptive_time_list_aux))[int(0.05 * NUM_EXPERIMENTS) : int(0.95 * NUM_EXPERIMENTS)])
  budget_mean = np.mean(np.sort(np.array(budget_list_aux))[int(0.05 * NUM_EXPERIMENTS) : int(0.95 * NUM_EXPERIMENTS)])


  print("=>delta:", imp_delta)
  print("=>average budget:", budget_mean)
  print("=>average error:", imp_epsilon)

  print("=>wrong_approx_num:", wrong_approx_num)

  dimension_list.append(N_FEATURES)
  naive_time_list.append(naive_time_mean)
  adaptive_time_list.append(adaptive_time_mean)
  budget_list.append(budget_mean)