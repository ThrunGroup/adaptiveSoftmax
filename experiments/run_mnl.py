import os
import pandas as pd
import numpy as np
from experiments.runner import run
from experiments.plotter import plot_scaling, get_scaling_param, clean_singleton_np_array_columns, get_budget_and_success_rate

from mnl.mnl_constants import (
    MNL_SCALING_POINTS,
    MNL_WEIGHTS_DIR,
    MNL_XS_DIR,
    MNL_RESULTS_DIR,

    MNIST,
    EUROSAT,
)


DEFAULT_SEED = 42

def scaling_mnl(A, X, dataset, path_dir):
    max_d = X.shape[1]
    np.random.seed(DEFAULT_SEED)

    for curr_d in np.linspace(0, max_d, MNL_SCALING_POINTS+1)[1:].astype(int):
        indices = np.random.choice(max_d, curr_d, replace=False)
        _A = A[:, indices]
        _X = X[:, indices]

        model = "mnl"
        dataset = dataset
        path = f"{path_dir}/d={curr_d}"

        print(run(
            save_to=path,
            model=model,
            dataset=dataset,
            A=_A,
            X=_X[:500],
            multiplicative_error = 0.3,
            failure_probability = 0.01,
            noise_bound = None,
            use_true_sftm = False,
            use_tune = True,
            train_size = 100,
        ))
        
def run_mnl():
    for dataset in [MNIST, EUROSAT]:
        path = f"testing_{dataset}_out256_iter10.npz"
    A = np.load(f"{MNL_WEIGHTS_DIR}/{mnist_path}")['data']
    X = np.load(f"{MNL_XS_DIR}/{mnist_path}")['data']


    path_dir = f"{MNL_RESULTS_DIR}/mnist"
    os.makedirs(path_dir, exist_ok=True)
    scaling_mnl(A, X, "mnist", path_dir)  # TODO: be more robust

    dimensions, naive_budgets, budgets, success_rates = get_scaling_param(path_dir)



if __name__ == "__main__":
    
    A = np.load(f"{MNL_WEIGHTS_DIR}/testing_mnist_out256_iter10.npz")['data']
    X = np.load(f"{MNL_XS_DIR}/testing_mnist_out256_iter10.npz")['data']

    # for file in sorted(os.listdir("results")):
    #   data = pd.read_csv(os.path.join("results", file))
    #   data = clean_singleton_np_array_columns(data)
    #   print(get_budget_and_success_rate(data))
    

    dimensions, budgets, naive_budgets, success_rates = get_scaling_param("results")
    plot_scaling(dimensions, naive_budgets, budgets, success_rates, "mnl")
    
  
