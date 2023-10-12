import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import ssl
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptive_softmax'))
from adasoftmax import ada_softmax

# TODO(@lukehan): Change indentation

if __name__ == "__main__":
    torch.manual_seed(777)
    np.random.seed(777)

    device = torch.devide('cuda:0') if torch.cuda.is_available else torch.device('cpu')

    ssl._create_default_https_context = ssl._create_unverified_context

def train_base_model(dataloader, model, device, max_iter=10):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(max_iter):
      avg_loss = 0

      for data, labels in dataloader:
        data.to(device)
        labels.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print(loss)

        avg_loss += loss / 256

      print(f"epoch {epoch} => loss: {avg_loss}")


def test_accuracy(test_loader, model, device):

    model.eval()
    accuracy = 0.0
    total = 0.0
    n_batches = 0

    with torch.no_grad():
        for data in test_loader:
            n_batches += 1
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            # run the model on the test set to predict labels
            prediction = model(images)
            # select the label with the highest logit
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

  # TODO(@lukehan): Might not need this. Test and remove.
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

  def transform_single(self, x):
      #assume batch is given
      with torch.no_grad():
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = torch.flatten(x)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
      return x

  def get_prob(self, x):
      with torch.no_grad():
        x = self.forward(x)
        return torch.nn.functional.softmax(x)

  def get_linear_weight(self):
      return self.linear2.weight.detach()

  def set_linear_weight(self, weight):
      self.linear.weight = torch.nn.parameter.Parameter(weight)


class TransformToLinear(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device
    def __call__(self, image):
        image = self.model.transform_single(image)
        image = image.to(device)

        return image


EuroSAT = datasets.EuroSAT(root="./data/",
                                 download=True,
                                 transform=transforms.ToTensor())

train_set_indices = np.random.choice(27000, size=(21600,), replace=False)
tmp = np.ones(27000)
tmp[train_set_indices] = 0
test_set_indices = np.nonzero(tmp)

train_dataloader = DataLoader(Subset(EuroSAT, train_set_indices), batch_size=256, shuffle=True)
base_model = EuroSATModel().to(device)
train_base_model(train_dataloader, base_model, max_iter=2)

base_model.eval()


test_dataloader = DataLoader(Subset(EuroSAT, test_set_indices[0]), batch_size=256, shuffle=False)
print(test_accuracy(test_dataloader, base_model))

flatten_transform = transforms.Compose([transforms.ToTensor(),
                                        TransformToLinear(base_model, device)])
EuroSAT_flattened = datasets.EuroSAT(root="./data/",
                                 download=True,
                                 transform=flatten_transform)
test_set_flattened = Subset(EuroSAT_flattened, test_set_indices[0])
test_set_flattened_loader = DataLoader(test_set_flattened, batch_size=1, shuffle=False)

# TODO(@lukehan): Move this to the plotting section
plt.rcParams['figure.figsize'] = [6.4, 4.8]


dimension_list = list()
budget_list = list()
sigma_list = list()
gain_list = list()
delta_list = list()
error_list = list()


TEMP = 1
N_CLASSES = 10
NUM_EXPERIMENTS = 100


for dimension in list(range(25600, 25600 + 1, 1000)):
  print("dimension:", dimension)
  dimension_list.append(dimension)
  budget_list_aux = list()

  wrong_approx_num = 0
  budget_sum = 0
  error_sum = 0
  gain_sum = 0

  # Extract linear layer's weight from tranied model
  A = base_model.get_linear_weight()
  A_ndarray = A.detach().cpu().numpy()

  epsilon = 0.1
  delta = 0.01
  top_k = 1

  for seed in range(NUM_EXPERIMENTS):
    print(seed)
    x, label = test_set_flattened_loader[seed]
    x_ndarray = x[0].detach().numpy()

    # naive softmax
    mu = A_ndarray @ x_ndarray
    mu_exp = np.exp(mu - np.max(mu))
    z = mu_exp / np.sum(mu_exp)


    mu = A_ndarray @ x_ndarray
    print("gain:", N_CLASSES * np.sum(np.exp(2 * (mu - np.max(mu)))) / (np.sum(np.exp(mu - np.max(mu)))**2))
    gain_sum += N_CLASSES * np.sum(np.exp(2 * (mu - np.max(mu)))) / (np.sum(np.exp(mu - np.max(mu)))**2)

    # AdaSoftmax
    bandit_topk_indices, z_hat, bandit_budget = ada_softmax(A_ndarray, x_ndarray, TEMP, epsilon, delta, dimension, top_k)

    # TODO(@lukehan): Change how we evaluate delta and epsilon(Like in the adaptive_softmax/test_script.py)
    cur_epsilon = np.abs(z_hat[bandit_topk_indices] - z[bandit_topk_indices]) / z[bandit_topk_indices]
    print(z_hat[bandit_topk_indices], z[bandit_topk_indices])

    if cur_epsilon[0] > 1e-2:
      print(cur_epsilon)

    if cur_epsilon[0] <= epsilon and bandit_topk_indices[0] == np.argmax(z): #ASSUMING K=1
      error_sum += cur_epsilon[0]
    elif bandit_topk_indices[0] == np.argmax(z):
      wrong_approx_num += 1
      error_sum += cur_epsilon[0]
    else:
      print(seed)
      #error_sum += cur_epsilon[0]
      wrong_approx_num += 1
      print(bandit_budget)
      print(z)
      print(z_hat)
      print(label, bandit_topk_indices[0], np.argmax(z), cur_epsilon[0])

    budget_list_aux.append(bandit_budget)

  imp_delta = wrong_approx_num / NUM_EXPERIMENTS
  average_budget = budget_sum / NUM_EXPERIMENTS
  # TODO(@lukehan): Take median instead
  imp_epsilon = error_sum / NUM_EXPERIMENTS
  gain_mean = gain_sum / NUM_EXPERIMENTS
  budget_mean = np.mean(budget_list_aux)


  print("=>delta:", imp_delta)
  print("=>average budget:", budget_mean)
  print("=>average error:", imp_epsilon)

  print("=>wrong_approx_num:", wrong_approx_num)

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