import torch
import numpy as np
import ssl

from typing import Tuple, Iterator

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from .model import BaseModel, TransformToLinear
from .mnl_constants import *

def train_base_model(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,   # TODO: should this be str?
    max_iter: int = TRAINING_ITERATIONS,
    verbose: bool = False,
) -> None:
    """
    Trains base model.
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

            avg_loss += loss / BATCH_SIZE

        if verbose:
            print(f"epoch {epoch} => loss: {avg_loss}")


def test_accuracy(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Tests base model.
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

def get_A_and_x(dataset: str) -> Tuple[np.ndarray, Iterator]:
    """
    Trains the basemodel on the dataset for which we will extract matrix A and vector x.
    You can set TRAINING_ITERATIONS = 0 if you want this to be completely random. 

    :returns: (matrix A, iterator for x)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataset and params
    if dataset == MNIST:
        in_channel = MNIST_IN_CHANNEL
        out_channel = MNIST_OUT_CHANNEL
        in_feature = MNIST_IN_FEATURE
        path = MNIST_PATH
        training_set = datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(), download=True)

    elif dataset == EUROSAT:
        # need this certificate to download
        ssl._create_default_https_context = ssl._create_unverified_context
        
        in_channel = EUROSAT_IN_CHANNEL
        out_channel = EUROSAT_OUT_CHANNEL
        in_feature = EUROSAT_IN_FEATURE
        path = EUROSAT_PATH
        training_set = datasets.EuroSAT(root=path, transform=transforms.ToTensor(), download=True)

        # separate test set doesn't exist so subsampling
        train_size = EUROSAT_DATAPOINTS - NUM_EXPERIMENTS
        train_indices = np.random.choice(EUROSAT_DATAPOINTS, size=(train_size,), replace=False)
        training_set = Subset(training_set, train_indices)
        x_indices = np.setdiff1d(np.arange(EUROSAT_DATAPOINTS), train_indices)

    # train the model
    dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    model = BaseModel(in_channel, out_channel, in_feature).to(device)
    train_base_model(dataloader, model, device)

    # extract A and x (NOTE: x is the testing datapoints)
    get_x = transforms.Compose([
        transforms.ToTensor(),
        TransformToLinear(model, device)
    ])
    if dataset == MNIST:
        xs = datasets.MNIST(root=path, train=False, transform=get_x, download=True)

    elif dataset == EUROSAT:
        xs = datasets.EuroSAT(root=path, transform=get_x, download=True)
        xs = Subset(xs, x_indices)

    x_loader = DataLoader(xs, batch_size=1, shuffle=False)
    iterator = iter(x_loader)

    A = model.get_linear_weight()
    A = A.detach().cpu().numpy()

    return (A, iterator)

    