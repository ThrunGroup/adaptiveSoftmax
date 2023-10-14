import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import ssl
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptive_softmax'))
from adasoftmax import ada_softmax, approx_sigma
from hadamard_transform import hadamard_transform
from MNIST_constants import CONV_OUTPUT_CHANNEL, LINEAR_DIMENSION, TEMP
from constants import PROFILE

# TODO(@lukehan): Move constants to seperate file


if __name__ == "__main__":
    torch.manual_seed(777)
    np.random.seed(777)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ssl._create_default_https_context = ssl._create_unverified_context

def train_base_model(dataloader, model, device, max_iter=10):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(max_iter):
      avg_loss = 0

      for data, labels in dataloader:
        data = data.to(device)
        labels = labels.to(device)

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


class BaseModel(torch.nn.Module):
  def __init__(self, num_output_channel):
        super().__init__()

        self.conv = torch.nn.Conv2d(1, num_output_channel, 3)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.linear = torch.nn.Linear(13*13*num_output_channel, 10, bias=False, dtype=torch.float)
        self.dropout = torch.nn.Dropout(0.25)

  def forward(self, x):
      x = self.conv(x)
      x = torch.nn.functional.relu(x)
      x = self.pool(x)
      x = self.dropout(x)
      x = torch.flatten(x, start_dim=1)
      out = self.linear(x)
      return out

  def transform_single(self, x):
      #assume batch is given
      with torch.no_grad():
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        out = torch.flatten(x)
      return out

  def get_prob(self, x):
      with torch.no_grad():
        x = self.forward(x)
        return torch.nn.functional.softmax(x)

  def get_linear_weight(self):
      return self.linear.weight.detach()

  def set_linear_weight(self, weight):
      self.linear.weight = torch.nn.parameter.Parameter(weight)


class TransformToLinear(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device
    def __call__(self, image):
        image = image.to(device)
        image = self.model.transform_single(image)

        return image


MNIST_train = datasets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

MNIST_test = datasets.MNIST(root='MNIST_data/',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)


train_dataloader = DataLoader(MNIST_train, batch_size=256, shuffle=True)


test_dataloader = DataLoader(MNIST_test, batch_size=256, shuffle=False)

# TODO(@lukehan): Move this to the plotting section
plt.rcParams['figure.figsize'] = [6.4, 4.8]


dimension_list = list()
budget_list = list()
sigma_list = list()
gain_list = list()
delta_list = list()
error_list = list()
accuracy_list = list()

use_hadamard_transform = False

N_CLASSES = 10
NUM_EXPERIMENTS = 100


for num_output_channel in range(32, 76, 4):
    base_model = BaseModel(num_output_channel).to(device)
    train_base_model(train_dataloader, base_model, device, max_iter=1)
    base_model.eval()
    accuracy = test_accuracy(test_dataloader, base_model, device).item()
    accuracy_list.append(accuracy)
    flatten_transform = transforms.Compose([transforms.ToTensor(),
                                            TransformToLinear(base_model, device)])
    MNIST_test_flattened = datasets.MNIST(root='MNIST_data/',
                                          train=False,
                                          transform=flatten_transform,
                                          download=True)
    test_set_flattened_loader = DataLoader(MNIST_test_flattened, batch_size=1, shuffle=False)

    wrong_approx_num = 0
    budget_sum = 0
    error_sum = 0
    gain_sum = 0
    sigma_sum = 0

    # Extract linear layer's weight from tranied model
    A = base_model.get_linear_weight()
    A_ndarray = A.detach().cpu().numpy()

    dimension = A.shape[1]

    print("dimension:", dimension)
    dimension_list.append(dimension)
    budget_list_aux = list()

    epsilon = 0.1
    delta = 0.01
    top_k = 1

    if use_hadamard_transform:
        dPad = int(2 ** np.ceil(np.log2(dimension)))
        D = np.diag(np.random.choice([-1, 1], size=dPad))

    test_set_iterator = iter(test_set_flattened_loader)

    for seed in range(NUM_EXPERIMENTS):
        #print(seed)
        x, label = next(test_set_iterator)
        x_ndarray = x[0].detach().cpu().numpy()

        # naive softmax
        mu = TEMP * A_ndarray @ x_ndarray
        mu_exp = np.exp(mu - np.max(mu))
        z = mu_exp / np.sum(mu_exp)

        gain = N_CLASSES * np.sum(np.exp(2 * (mu - np.max(mu)))) / (np.sum(np.exp(mu - np.max(mu)))**2)
        #print("gain:", gain)
        gain_sum += gain

        sigma = approx_sigma(A_ndarray, x_ndarray, dimension, TEMP)
        sigma_sum += sigma


        # AdaSoftmax
        if use_hadamard_transform:
            Apad = np.pad(A_ndarray, ((0, 0), (0, dPad - dimension)), 'constant', constant_values=0)
            xpad = np.pad(x_ndarray, (0, dPad - dimension), 'constant', constant_values=0)

            # convert padded A and x to Tensor in order to use pytorch's hadamard transform library
            A_pad_torch = torch.tensor(Apad)
            x_pad_torch = torch.tensor(xpad)

            Aprime = hadamard_transform(A_pad_torch @ D).numpy()
            xprime = hadamard_transform(x_pad_torch @ D).numpy()

            bandit_topk_indices, z_hat, bandit_budget = ada_softmax(A=Aprime,
                                                                    x=xprime,
                                                                    epsilon=epsilon,
                                                                    delta=delta,
                                                                    samples_for_sigma=dPad,
                                                                    beta=TEMP,
                                                                    k=top_k,
                                                                    verbose=True
                                                                    )
        elif PROFILE:
            bandit_topk_indices, z_hat, bandit_budget, profiling_results = ada_softmax(A=A_ndarray,
                                                                                      x=x_ndarray,
                                                                                      epsilon=epsilon,
                                                                                      delta=delta,
                                                                                      samples_for_sigma=dimension,
                                                                                      beta=TEMP,
                                                                                      k=top_k,
                                                                                      verbose=True,
                                                                                      )

            for (key, value) in profiling_results.items():
                print(key, value)
        else:
            bandit_topk_indices, z_hat, bandit_budget = ada_softmax(A=A_ndarray,
                                                                    x=x_ndarray,
                                                                    epsilon=epsilon,
                                                                    delta=delta,
                                                                    samples_for_sigma=dimension,
                                                                    beta=TEMP,
                                                                    k=top_k,
                                                                    verbose=False
                                                                    )

        # TODO(@lukehan): Change how we evaluate delta and epsilon(Like in the adaptive_softmax/test_script.py)
        cur_epsilon = np.abs(z_hat[bandit_topk_indices] - z[bandit_topk_indices]) / z[bandit_topk_indices]
        # print(z_hat[bandit_topk_indices], z[bandit_topk_indices])


        if cur_epsilon[0] <= epsilon and bandit_topk_indices[0] == np.argmax(z): #ASSUMING K=1
            error_sum += cur_epsilon[0]
        elif bandit_topk_indices[0] == np.argmax(z):
            wrong_approx_num += 1
        else:
            print(seed)
            #error_sum += cur_epsilon[0]
            wrong_approx_num += 1
            print(bandit_budget)
            print(z)
            print(z_hat)
            print(label, bandit_topk_indices[0], np.argmax(z), cur_epsilon[0])

        budget_sum += bandit_budget

    imp_delta = wrong_approx_num / NUM_EXPERIMENTS
    average_budget = budget_sum / NUM_EXPERIMENTS
    # TODO(@lukehan): Take median instead
    imp_epsilon = error_sum / NUM_EXPERIMENTS
    gain_mean = gain_sum / NUM_EXPERIMENTS
    sigma_mean = sigma_sum / NUM_EXPERIMENTS

    print("=>delta:", imp_delta)
    print("=>average budget:", average_budget)
    print("=>Naive budget is:", dimension*N_CLASSES)
    print("=>average error:", imp_epsilon)
    print("=>average gain:", gain_mean)
    print("=>average sigma:", sigma_mean)

    print("=>wrong_approx_num:", wrong_approx_num)

    budget_list.append(average_budget)
    gain_list.append(gain_mean)
    error_list.append(imp_epsilon)
    delta_list.append(imp_delta)
    sigma_list.append(sigma_mean)

dimension_list = np.array(dimension_list)
budget_list = np.array(budget_list)

# Linear fit on adaSoftmax's sample complexity for cleaner plot
c1, c0 = np.polyfit(dimension_list, budget_list, 1)
linear_fit_points = c1 * dimension_list + c0

# Sample complexity plot
plt.plot(dimension_list, linear_fit_points, "r--.", label="adaptive linear fit")
plt.scatter(dimension_list, budget_list, color="red", label="adaptive_softmax")
plt.plot(dimension_list, N_CLASSES * dimension_list, "b--.", label="naive")
plt.legend()
plt.xlabel("dimension(n_features)")
plt.ylabel("number of samples taken")
plt.savefig("sample_complexity_plot.svg", bbox_inches="tight")
plt.clf()

# Sample complexity plot(on log scale)
plt.yscale("log")
plt.plot(dimension_list, budget_list, "r--.", label="adaptive_softmax")
plt.plot(dimension_list, N_CLASSES * dimension_list, "b--.", label="naive")
plt.legend()
plt.xlabel("dimension(n_features)")
plt.ylabel("number of samples taken")
plt.savefig("sample_complexity_log_plot.svg", bbox_inches="tight")
plt.clf()

# accuracy plot
plt.plot(dimension_list, accuracy_list)
plt.xlabel("dimension")
plt.ylabel("accuracy")
plt.savefig("accuracy_plot.png", bbox_inches="tight")
plt.clf()

# average gain plot
plt.plot(dimension_list, gain_list)
plt.xlabel("dimension")
plt.ylabel("average gain")
plt.savefig("gain_plot.png", bbox_inches="tight")
plt.clf()

# average sigma(median) plot
plt.plot(dimension_list, sigma_list)
plt.xlabel("dimension")
plt.ylabel("average sigma")
plt.savefig("sigma_plot.png", bbox_inches="tight")
