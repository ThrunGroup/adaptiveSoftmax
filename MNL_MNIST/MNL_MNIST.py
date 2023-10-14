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
from MNIST_constants import TEMP, EPSILON, DELTA, NUM_EXPERIMENTS, USE_HADAMARD_TRANSFORM, TOP_K


if __name__ == "__main__":
    torch.manual_seed(777)
    np.random.seed(777)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Code to enable data download
    ssl._create_default_https_context = ssl._create_unverified_context

def train_base_model(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    max_iter: int = 10,
    verbose: bool = False,
) -> None:
    """
    Trains the model with the training data given by the dataloader.

    :param dataloader: Pytorch dataloader for training data
    :param model: MNL model to train
    :param device: Device to train model on
    :param max_iter: Training iteration
    :param verbose: Indicator for verbosity
    :return: None
    """
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

            avg_loss += loss / 256

        if verbose:
            print(f"epoch {epoch} => loss: {avg_loss}")


def test_accuracy(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Test the model's accuracy with the given test data
    :param test_loader: Pytorch Dataloader for test data
    :param model: Model to test the accuracy
    :param device: Device to run the model on
    :return: accuracy of the given model on the given test data
    """
    model.eval()
    accuracy = 0.0
    n_batches = 0

    with torch.no_grad():
        for data in test_loader:
            n_batches += 1
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            prediction = model(images)
            correct_prediction = torch.argmax(prediction, 1) == labels
            accuracy += correct_prediction.float().mean()

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
        """
        Transforms the given raw data in a flattened 1D vector form.
        This passes the raw image through the forward pass of the model until the linear layer so that
        when the resulting transformed vector x' is multiplied with A, it's equivalent of passing x through
        the model.
        :param x: raw data
        :return: transformed 1D vector
        """
        with torch.no_grad():
            x = self.conv(x)
            x = torch.nn.functional.relu(x)
            x = self.pool(x)
            out = torch.flatten(x)
        return out

    def get_linear_weight(self):
        return self.linear.weight.detach()


class TransformToLinear(object):
    """
    Pytorch Transform to transform raw data x to flattened 1D vector x'
    so that calculating A@x' is equivalent to passing x through the forward pass of the model.
    """
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


# List for test statistics on different dimensions.
dimension_list = list()
budget_list = list()
sigma_list = list()
gain_list = list()
delta_list = list()
error_list = list()
accuracy_list = list()

N_CLASSES = 10
NUM_EXPERIMENTS = 100


for num_output_channel in range(32, 76, 4):
    # Define MNL model, train on training set
    base_model = BaseModel(num_output_channel).to(device)
    train_base_model(train_dataloader, base_model, device, max_iter=1)

    # Evaluate test accuracy
    base_model.eval()
    accuracy = test_accuracy(test_dataloader, base_model, device).item()
    accuracy_list.append(accuracy)

    # Transform test data to a 1D vector.
    flatten_transform = transforms.Compose([transforms.ToTensor(),
                                            TransformToLinear(base_model, device)])
    MNIST_test_flattened = datasets.MNIST(root='MNIST_data/',
                                          train=False,
                                          transform=flatten_transform,
                                          download=True)
    test_set_flattened_loader = DataLoader(MNIST_test_flattened, batch_size=1, shuffle=False)

    # Variables for aggregating test statistics across all experiments
    wrong_approx_num = 0
    budget_sum = 0
    error_sum = 0
    gain_sublist = list()
    sigma_sublist = list()

    # Extract linear layer's weight from trained model
    A = base_model.get_linear_weight()
    A_ndarray = A.detach().cpu().numpy()

    # Get dimension of the linear layer
    dimension = A.shape[1]
    print("dimension:", dimension)
    dimension_list.append(dimension)

    if USE_HADAMARD_TRANSFORM:
        dPad = int(2 ** np.ceil(np.log2(dimension)))
        D = np.diag(np.random.choice([-1, 1], size=dPad))

    test_set_iterator = iter(test_set_flattened_loader)

    for seed in range(NUM_EXPERIMENTS):
        x, label = next(test_set_iterator)
        x_ndarray = x[0].detach().cpu().numpy()

        # naive softmax
        mu = TEMP * A_ndarray @ x_ndarray
        mu_exp = np.exp(mu - np.max(mu))
        z = mu_exp / np.sum(mu_exp)
        true_best_index = np.argmax(z)

        gain = N_CLASSES * np.sum(np.exp(2 * (mu - np.max(mu)))) / (np.sum(np.exp(mu - np.max(mu)))**2)
        gain_sublist.append(gain)

        sigma = approx_sigma(A_ndarray, x_ndarray, dimension, TEMP)
        sigma_sublist.append(sigma)


        # AdaSoftmax
        if USE_HADAMARD_TRANSFORM:
            # Pad A and x to the nearest power of 2 larger than d to enable hadamard transform
            Apad = np.pad(A_ndarray, ((0, 0), (0, dPad - dimension)), 'constant', constant_values=0)
            xpad = np.pad(x_ndarray, (0, dPad - dimension), 'constant', constant_values=0)

            # convert padded A and x to Tensor in order to use pytorch's hadamard transform library
            A_pad_torch = torch.tensor(Apad)
            x_pad_torch = torch.tensor(xpad)

            # Apply hadamard transform
            Aprime = hadamard_transform(A_pad_torch @ D).numpy()
            xprime = hadamard_transform(x_pad_torch @ D).numpy()

            bandit_topk_indices, z_hat, bandit_budget = ada_softmax(A=Aprime,
                                                                    x=xprime,
                                                                    epsilon=EPSILON,
                                                                    delta=DELTA,
                                                                    samples_for_sigma=dPad,
                                                                    beta=TEMP,
                                                                    k=TOP_K,
                                                                    verbose=True
                                                                    )
        else:
            bandit_topk_indices, z_hat, bandit_budget = ada_softmax(A=A_ndarray,
                                                                    x=x_ndarray,
                                                                    epsilon=EPSILON,
                                                                    delta=DELTA,
                                                                    samples_for_sigma=dimension,
                                                                    beta=TEMP,
                                                                    k=TOP_K,
                                                                    verbose=False
                                                                    )


        z_hat_best_arm = z_hat[bandit_topk_indices]
        z_best_arm = z[bandit_topk_indices]
        cur_epsilon = np.abs(z_hat_best_arm- z_best_arm) / z_best_arm

        if cur_epsilon[0] <= EPSILON and bandit_topk_indices[0] == true_best_index: #ASSUMING K=1
            error_sum += cur_epsilon[0]
        else:
            wrong_approx_num += 1

        budget_sum += bandit_budget

    # Aggregate test statistics across test of NUM_EXPERIMENT times
    imp_delta = wrong_approx_num / NUM_EXPERIMENTS
    average_budget = budget_sum / NUM_EXPERIMENTS
    # TODO(@lukehan): Take median instead
    imp_epsilon = error_sum / NUM_EXPERIMENTS
    gain_median = np.median(np.array(gain_sublist))
    sigma_median = np.median(np.array(sigma_sublist))

    print("=>delta:", imp_delta)
    print("=>average budget:", average_budget)
    print("=>Naive budget is:", dimension*N_CLASSES)
    print("=>average error:", imp_epsilon)
    print("=>median gain:", gain_median)
    print("=>median sigma:", sigma_median)
    print("=>wrong_approx_num:", wrong_approx_num)

    budget_list.append(average_budget)
    gain_list.append(gain_median)
    error_list.append(imp_epsilon)
    delta_list.append(imp_delta)
    sigma_list.append(sigma_median)

dimension_list = np.array(dimension_list)
budget_list = np.array(budget_list)

# Linear fit on adaSoftmax's sample complexity for cleaner plot
complexity_c1, complexity_c0 = np.polyfit(dimension_list, budget_list, 1)
complexity_linear_fit_points = complexity_c1 * dimension_list + complexity_c0

# Sample complexity plot
plt.plot(dimension_list, complexity_linear_fit_points, "r--.", label="adaptive linear fit")
plt.scatter(dimension_list, budget_list, color="red", label="adaptive_softmax")
plt.plot(dimension_list, N_CLASSES * dimension_list, "b--.", label="naive")
plt.legend()
plt.xlabel("dimension(n_features)")
plt.ylabel("number of samples taken")
plt.savefig("sample_complexity_plot.svg", bbox_inches="tight")
plt.clf()

# Sample complexity plot(on log scale)
plt.yscale("log")
plt.plot(dimension_list, complexity_linear_fit_points, "r--.", label="adaptive linear fit")
plt.scatter(dimension_list, budget_list, color="red", label="adaptive_softmax")
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

# linear fit on sigma for cleaner plot
sigma_c1, sigma_c0 = np.polyfit(dimension_list, sigma_list, 1)
sigma_linear_fit_points = sigma_c1 * dimension_list + sigma_c0

# average sigma(median) plot
plt.scatter(dimension_list, sigma_list)
plt.plot(dimension_list, sigma_linear_fit_points)
plt.xlabel("dimension")
plt.ylabel("average sigma")
plt.savefig("sigma_plot.png", bbox_inches="tight")
