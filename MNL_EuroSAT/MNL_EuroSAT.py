import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
import numpy as np
from numba import njit
import ssl
from ..adaptive_softmax.adasoftmax import ada_softmax

torch.manual_seed(777)
np.random.seed(777)

ssl._create_default_https_context = ssl._create_unverified_context


def train_base_model(dataloader, model, max_iter=10):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(max_iter):
      avg_loss = 0

      for data, labels in dataloader:
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print(loss)

        avg_loss += loss / 256

      print(f"epoch {epoch} => loss: {avg_loss}")


def test_accuracy(test_loader, model):

    model.eval()
    accuracy = 0.0
    total = 0.0
    n_batches = 0

    with torch.no_grad():
        for data in test_loader:
            n_batches += 1
            images, labels = data
            # run the model on the test set to predict labels
            prediction = model(images)
            # the label with the highest energy will be our prediction
            """
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            """
            correct_prediction = torch.argmax(prediction, 1) == labels
            accuracy += correct_prediction.float().mean()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / n_batches)
    return accuracy


class EuroSATModel(torch.nn.Module):
  def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3)
        self.pool = torch.nn.MaxPool2d(3, stride=3)
        self.linear1 = torch.nn.Linear(25600, 12800, dtype=torch.float)
        self.linear2 = torch.nn.Linear(12800, 10, bias=False, dtype=torch.float)  # Probably too drastic?
        self.dropout = torch.nn.Dropout(0.25)

  def forward(self, x):
      x = self.conv(x)
      x = torch.nn.functional.relu(x)
      x = self.pool(x)
      x = self.dropout(x)
      x = torch.flatten(x, start_dim=1)
      x = self.linear1(x)
      x = torch.nn.functional.relu(x)
      out = self.linear2(x)
      return out

  def matmul(self, x):
      with torch.no_grad():
          out = self.linear(x)
      return out

  def transform(self, X):
      #assume batch is given
      new_X = list()
      with torch.no_grad():
        for x in X:
            x = self.conv(x)
            x = torch.nn.functional.relu(x)
            x = self.pool(x)
            x = torch.flatten(x)
            x = self.linear1(x)
            x = torch.nn.functional.relu(x)
            new_X.append(x)
      return torch.stack(new_X).float()

  def get_prob(self, x):
      with torch.no_grad():
        x = self.forward(x)
        return torch.nn.functional.softmax(x)

  def get_linear_weight(self):
      return self.linear2.weight.detach()

  def set_linear_weight(self, weight):
      self.linear.weight = torch.nn.parameter.Parameter(weight)


EuroSAT_Train = datasets.EuroSAT(root="./data/",
                                 download=True,
                                 transform=transforms.ToTensor())

train_set_indices = np.random.choice(27000, size=(21600,), replace=False)
tmp = np.ones(27000)
tmp[train_set_indices] = 0
test_set_indices = np.nonzero(tmp)

train_dataloader = DataLoader(Subset(EuroSAT_Train, train_set_indices), batch_size=256, shuffle=True)

base_model = EuroSATModel()

train_base_model(train_dataloader, base_model, max_iter=10)

base_model.eval()


test_dataloader = DataLoader(Subset(EuroSAT_Train, test_set_indices[0]), batch_size=256, shuffle=False)


print(test_accuracy(test_dataloader, base_model))

test_set = Subset(EuroSAT_Train, train_set_indices)

new_test_data = [x for x, _ in test_set[:1000]]
new_test_labels = [y for _, y in test_set[:1000]]

new_test_data_flattened = base_model.transform(new_test_data)



def train(data, labels, model, max_iter=100):
    #print("training")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=0.9)

    for epoch in range(max_iter):
        print(epoch)
        running_loss = 0.0

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        print(loss / data.shape[0])


import torch
torch.manual_seed(0)

import numpy as np
from time import time
import matplotlib.pyplot as plt
from numba import njit
from torchvision import datasets, transforms
import ssl
import torch.functional as F
from torch.utils.data import DataLoader

plt.rcParams['figure.figsize'] = [6.4, 4.8]

class SimpleModel(torch.nn.Module):
    def __init__(self, temperature, data_dim, n_class):
        # simple classification model --> linear layer + softmax
        super().__init__()
        self.temperature = temperature
        self.linear = torch.nn.Linear(data_dim, n_class, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):   # TODO: add torch.no_grad()
        x = x * self.temperature
        x = self.linear(x)
        out = self.softmax(x)
        return out

    def matmul(self, x):
        with torch.no_grad():
            out = self.linear(x)
        return out

    def transform(self, x):
        return self.relu(self.linear(x))

    def get_linear_weight(self):
        return self.linear.weight.detach()

    def set_linear_weight(self, weight):
        self.linear.weight = torch.nn.parameter.Parameter(weight)

dimension_list = list()
naive_time_list = list()
adaptive_time_list = list()
budget_list = list()
sigma_list = list()
gain_list = list()
delta_list = list()
error_list = list()


TEMP = 1
N_CLASSES = 10
NUM_EXPERIMENTS = 1000

#new_data, new_labels = split_data(EuroSAT_Train)

#new_test_data, new_test_labels = new_data[random_subset], new_labels[random_subset]

#print("data prepped")

for dimension in list(range(12800, 12800 + 1, 1000)):
  print("dimension:", dimension)
  dimension_list.append(dimension)
  naive_time_list_aux = list()
  adaptive_time_list_aux = list()
  budget_list_aux = list()

  wrong_approx_num = 0
  budget_sum = 0
  error_sum = 0
  time_dif_sum = 0
  naive_time_sum = 0
  adaptive_time_sum = 0
  gain_sum = 0

  #new_data = torch.cat(tuple([data]*dim_constant), 1)
  #subsample_index = np.random.choice(data[0].shape[0], size=dimension, replace=False)
  #new_data = data[:, subsample_index].detach()
  #model = SimpleModel(TEMP, dimension, 10)
  #model = EuroSATModel()
  #train(new_data, new_labels, model, max_iter=20)
  #convolutions = model.get_convolutions()

  A = base_model.get_linear_weight()
  A_ndarray = A.detach().numpy()
  #new_A = A[:, subsample_index]
  #A_ndarray = new_A.detach().numpy()

  sigma_array = np.empty((NUM_EXPERIMENTS, 10))



  epsilon = 0.1
  delta = 0.01
  top_k = 1

  for seed in range(NUM_EXPERIMENTS):
    print(seed)
    x = new_test_data_flattened[seed]
    x_ndarray = x.detach().numpy()

    # naive softmax
    naive_start_time = time()
    #z = model(x)
    mu = A_ndarray @ x_ndarray
    mu_exp = np.exp(mu - np.max(mu))
    z = mu_exp / np.sum(mu_exp)
    naive_time = time() - naive_start_time
    naive_time_sum += naive_time

    #TEST
    naive_time_list_aux.append(naive_time)

    #x_ndarray = model.transform(x).detach().numpy()


    mu = A_ndarray @ x_ndarray
    print("gain:", N_CLASSES * np.sum(np.exp(2 * (mu - np.max(mu)))) / (np.sum(np.exp(mu - np.max(mu)))**2))
    gain_sum += N_CLASSES * np.sum(np.exp(2 * (mu - np.max(mu)))) / (np.sum(np.exp(mu - np.max(mu)))**2)

    # AdaSoftmax

    adaptive_start_time = time()
    bandit_topk_indices, z_hat, bandit_budget, sigma = ada_softmax_nb(A_ndarray, x_ndarray, TEMP, epsilon, delta, dimension, top_k)
    adaptive_time = time() - adaptive_start_time
    adaptive_time_sum += adaptive_time

    sigma_array[seed] = sigma

    adaptive_time_list_aux.append(adaptive_time)

    #numpy_z = z.detach().cpu().numpy()[new_test_labels[seed]]
    #numpy_z = z[labels]
    numpy_z = z

    cur_epsilon = np.abs(z_hat[bandit_topk_indices] - numpy_z[bandit_topk_indices]) / numpy_z[bandit_topk_indices]
    print(z_hat[bandit_topk_indices], numpy_z[bandit_topk_indices])

    if cur_epsilon[0] > 1e-2:
      print(cur_epsilon)

    if cur_epsilon[0] <= epsilon and bandit_topk_indices[0] == np.argmax(numpy_z): #ASSUMING K=1
      error_sum += cur_epsilon[0]
    elif bandit_topk_indices[0] == np.argmax(numpy_z):
      wrong_approx_num += 1
      error_sum += cur_epsilon[0]
    else:
      print(seed)
      #error_sum += cur_epsilon[0]
      wrong_approx_num += 1
      print(bandit_budget)
      print(z)
      print(z_hat)
      print(new_test_labels[seed], bandit_topk_indices[0], np.argmax(z), cur_epsilon[0])

    budget_list_aux.append(bandit_budget)

  imp_delta = wrong_approx_num / NUM_EXPERIMENTS
  average_budget = budget_sum / NUM_EXPERIMENTS
  imp_epsilon = error_sum / NUM_EXPERIMENTS
  gain_mean = gain_sum / NUM_EXPERIMENTS

  #naive_time_mean = np.mean(np.sort(np.array(naive_time_list_aux))[int(0.05 * NUM_EXPERIMENTS) : int(0.95 * NUM_EXPERIMENTS)])
  #adaptive_time_mean = np.mean(np.sort(np.array(adaptive_time_list_aux))[int(0.05 * NUM_EXPERIMENTS) : int(0.95 * NUM_EXPERIMENTS)])
  #budget_mean = np.mean(np.sort(np.array(budget_list_aux))[int(0.05 * NUM_EXPERIMENTS) : int(0.95 * NUM_EXPERIMENTS)])
  budget_mean = np.mean(budget_list_aux)


  print("=>delta:", imp_delta)
  print("=>average budget:", budget_mean)
  print("=>average error:", imp_epsilon)

  print("=>wrong_approx_num:", wrong_approx_num)

  #naive_time_list.append(naive_time_mean)
  #adaptive_time_list.append(adaptive_time_mean)
  budget_list.append(budget_mean)
  gain_list.append(gain_mean)
  error_list.append(imp_epsilon)
  delta_list.append(imp_delta)

dimension_list = np.array(dimension_list)
budget_list = np.array(budget_list)

plt.plot(dimension_list, budget_list, "r--.", label="adaptive_softmax")
plt.plot(dimension_list, N_CLASSES * dimension_list, "b--.", label="naive")
plt.legend()
plt.xlabel("dimension(n_features)")
plt.ylabel("number of samples taken")
plt.savefig("sample_complexity_plot.svg", bbox_inches="tight")
plt.yscale("log")
plt.savefig("sample_complexity_log_plot.svg", bbox_inches="tight")