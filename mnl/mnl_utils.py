import os
import torch
import numpy as np
import pandas as pd
import ssl

from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from .model import BaseModel
from .train_and_eval import train, test
from .mnl_constants import *

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
        dataset = datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(), download=True)

        # split into train and val
        train_size = int(0.8 * len(dataset))
        train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
        test_set = datasets.MNIST(root=path, train=False, transform=transforms.ToTensor(), download=True)

    elif dataset == EUROSAT:
        # need this certificate to download
        ssl._create_default_https_context = ssl._create_unverified_context
        
        in_channel = EUROSAT_IN_CHANNEL
        out_channel = EUROSAT_OUT_CHANNEL
        path = EUROSAT_PATH
        dataset = datasets.EuroSAT(root=path, transform=transforms.ToTensor(), download=True)

        # split into train, val, test
        test_size = int(0.2 * len(dataset))
        train_val, test_set = random_split(dataset, [len(dataset) - test_size, test_size])
        train_size = int(0.8 * len(train_val))
        train_set, val_set = random_split(train_val, [train_size, len(train_val) - train_size])

    # get dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # train model
    model = BaseModel(in_channel, out_channel).to(device)
    train(train_loader, val_loader, model, device)
    acc = test(test_loader, model, device)

    # extract A and x 
    A = model.get_linear_weight().cpu().numpy()
    xs = model.extract_features(test_loader, device)
    return A, xs, acc


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
    os.makedirs(MNL_ACC_DIR, exist_ok=True)
    if dataset == MNIST:
        out_channel = MNIST_OUT_CHANNEL
    else:
        out_channel = EUROSAT_OUT_CHANNEL

    path = f'{dataset}_out{out_channel}_iter{train_iterations}.npz'
    if testing: 
        path = f'testing_{path}'

    weights_path = f'{MNL_WEIGHTS_DIR}/{path}'
    x_matrix_path = f'{MNL_XS_DIR}/{path}'

    # Check if the files exist
    if os.path.exists(weights_path) and os.path.exists(x_matrix_path):
        A = np.load(weights_path, allow_pickle=False)['data']
        x_matrix = np.load(x_matrix_path, allow_pickle=False)['data']
    else:
        A, x_matrix, acc = generate_A_and_x(dataset)
        pd.DataFrame({path: [acc]}).to_csv(f"{MNL_ACC_DIR}/{path[:-4]}.csv")
        np.savez_compressed(weights_path[:-4], data=A)
        np.savez_compressed(x_matrix_path[:-4], data=x_matrix)
    
    return A, x_matrix
