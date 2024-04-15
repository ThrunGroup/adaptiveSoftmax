import os
import torch
import numpy as np
import ssl

from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from .model import BaseModel
from .mnl_constants import *

def train_base_model(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,   # TODO: should this be str?
    max_iter: int = TRAINING_ITERATIONS,
    verbose: bool = True,
) -> None:
    """
    Trains base model.
    """
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(max_iter):
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(data)    # logits
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        if verbose:
            print(f"epoch {epoch} => loss: {loss}")


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


def generate_A_and_x(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains the basemodel on the dataset for which we will extract matrix A and vector x.
    You can set TRAINING_ITERATIONS = 0 if you want this to be completely random. 

    :returns: (matrix A, iterator for x)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(MNL_TEST_SEED)
    torch.manual_seed(MNL_TEST_SEED)

    # get dataset and params
    if dataset == MNIST:
        in_channel = MNIST_IN_CHANNEL
        out_channel = MNIST_OUT_CHANNEL
        path = MNIST_PATH
        training_set = datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(), download=True)

    elif dataset == EUROSAT:
        # need this certificate to download
        ssl._create_default_https_context = ssl._create_unverified_context
        
        in_channel = EUROSAT_IN_CHANNEL
        out_channel = EUROSAT_OUT_CHANNEL
        path = EUROSAT_PATH
        training_set = datasets.EuroSAT(root=path, transform=transforms.ToTensor(), download=True)

        # separate test set doesn't exist so subsampling
        train_size = EUROSAT_DATAPOINTS - NUM_EXPERIMENTS
        train_indices = np.random.choice(EUROSAT_DATAPOINTS, size=(train_size,), replace=False)
        training_set = Subset(training_set, train_indices)
        x_indices = np.setdiff1d(np.arange(EUROSAT_DATAPOINTS), train_indices)

    # train the model and get test data (NOTE: x is from the test set)
    dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    model = BaseModel(in_channel, out_channel).to(device)
    train_base_model(dataloader, model, device)

    if dataset == MNIST:
        testset = datasets.MNIST(root=path, train=False, transform=transforms.ToTensor(), download=True)
    elif dataset == EUROSAT:
        testset = datasets.EuroSAT(root=path, transform=transforms.ToTensor(), download=True)
        testset = Subset(testset, x_indices)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # extract A and x 
    A = model.get_linear_weight().cpu().numpy()
    xs = model.extract_features(test_loader, device)
    return A, xs


def load_A_and_xs(
        dataset: str, 
        testing: bool = True,
        train_iterations: int = TRAINING_ITERATIONS,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads A matrix (weights) and x matrix (features) for a given dataset.
    If doesn't exist, will generate.
    
    :param dataset: expected values are 'MNIST' or 'EUROSAT'.
    :returns: A and x
    """
    # Ensure the directories exist
    os.makedirs(MNL_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(MNL_XS_DIR, exist_ok=True)
    if dataset == MNIST:
        out_channel = MNIST_OUT_CHANNEL
    else:
        out_channel = EUROSAT_OUT_CHANNEL

    path = f'{dataset}_out{out_channel}_iter{train_iterations}.npy'
    if testing: 
        path = f'testing_{path}'

    weights_path = f'{MNL_WEIGHTS_DIR}/{path}'
    x_matrix_path = f'{MNL_XS_DIR}/{path}'

    # Check if the files exist
    if os.path.exists(weights_path) and os.path.exists(x_matrix_path):
        A = np.load(weights_path)
        x_matrix = np.load(x_matrix_path)
    else:
        A, x_matrix = generate_A_and_x(dataset)
        np.save(weights_path, A)
        np.save(x_matrix_path, x_matrix)
    
    return A, x_matrix
