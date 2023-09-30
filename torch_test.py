# installing dependencies
import torch
torch.manual_seed(0)

from time import time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from adaSoftmax_torch import ada_softmax


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

  new_data = (1/dim_constant) * torch.cat(tuple([data]*dim_constant), 1)
  model = SimpleModel(TEMP, new_data.shape[1], N_CLASSES)
  train(new_data, labels, model, max_iter=30)
  A = model.get_linear_weight()


  epsilon = 0.2
  delta = 0.01
  top_k = 1

  for seed in range(NUM_EXPERIMENTS):
    x = new_data[seed % N_DATA]

    # naive softmax
    naive_start_time = time()
    z = model(x)
    naive_time = time() - naive_start_time
    naive_time_sum += naive_time

    #TEST
    naive_time_list_aux.append(naive_time)

    # AdaSoftmax

    adaptive_start_time = time()
    bandit_topk_indices, z_hat, bandit_budget = ada_softmax(A, x, TEMP, epsilon, delta, dim_constant * data.shape[1], top_k)
    adaptive_time = time() - adaptive_start_time
    adaptive_time_sum += adaptive_time

    adaptive_time_list_aux.append(adaptive_time)

    numpy_z = z.detach().numpy()[labels]

    cur_epsilon = torch.abs(z_hat[bandit_topk_indices] - torch.max(z)) / torch.max(z).item()

    if cur_epsilon[0] <= epsilon and bandit_topk_indices[0] == labels[seed % N_DATA]: #ASSUMING K=1
      error_sum += cur_epsilon[0]
    else:
      wrong_approx_num += 1

    budget_list_aux.append(bandit_budget)

  imp_delta = wrong_approx_num / NUM_EXPERIMENTS
  average_budget = budget_sum / NUM_EXPERIMENTS
  imp_epsilon = error_sum / NUM_EXPERIMENTS

  budget_mean = torch.mean(torch.Tensor(budget_list_aux))


  print("=>delta:", imp_delta)
  print("=>average budget:", budget_mean)
  print("=>average error:", imp_epsilon)

  print("=>wrong_approx_num:", wrong_approx_num)

  dimension_list.append(N_FEATURES)
  #naive_time_list.append(naive_time_mean)
  #adaptive_time_list.append(adaptive_time_mean)
  budget_list.append(budget_mean)

"""
plt.plot(12288 * np.arange(1, NUM_DIMENSIONS + 1), budget_list, "r--.", label="adaptive_softmax")
plt.plot(12288 * np.arange(1, NUM_DIMENSIONS + 1), N_CLASSES * 12288 * np.arange(1, NUM_DIMENSIONS + 1), "b--.", label="naive")
plt.legend()
plt.xlabel("dimension(n_features)")
plt.ylabel("number of samples taken")
plt.savefig("sample_complexity_plot.png")
"""