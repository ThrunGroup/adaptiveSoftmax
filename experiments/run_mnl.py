import os
import time
import numpy as np
from experiments.runner import run
from experiments.plotter import plot_scaling, get_scaling_param, clean_singleton_np_array_columns, get_budget_and_success_rate

from mnl.mnl_constants import (
    MNL_SCALING_POINTS,
    MNL_WEIGHTS_DIR,
    MNL_XS_DIR,
    MNL_RESULTS_DIR,

    MNIST,
    MNIST_FINAL_PATH,
    EUROSAT,
    EUROSAT_FINAL_PATH,
    SEED,
    NUM_QUERIES,

    DELTA,
    EPS
)


def scaling_mnl(A, X, dataset, path_dir, delta, eps):
    max_d = X.shape[1]
    np.random.seed(SEED)

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
            X=_X,
            multiplicative_error=eps,
            failure_probability=delta,
            noise_bound = None,
            use_true_sftm = False,
            use_tune = True,
            train_size = 100,
            exact_pull_best_arm=False  # this could work if variance is small???
        ))
        
def run_mnl(delta, eps):
    curr_time = time.strftime("%H:%M:%S", time.gmtime())
    for dataset in [MNIST, EUROSAT]:
        loading_path = MNIST_FINAL_PATH if dataset == MNIST else EUROSAT_FINAL_PATH

    A = np.load(f"{MNL_WEIGHTS_DIR}/{loading_path}")['data']
    X = np.load(f"{MNL_XS_DIR}/{loading_path}")['data']

    save_dir = f"{MNL_RESULTS_DIR}_{curr_time}/{dataset}/delta{delta}_eps{eps}"
    os.makedirs(save_dir, exist_ok=True)
    scaling_mnl(A, X[:NUM_QUERIES], "mnist", save_dir, delta, eps)  

    dimensions, naive_budgets, budgets, success_rates = get_scaling_param(save_dir)
    plot_scaling(dimensions, naive_budgets, budgets, success_rates, "mnl")


if __name__ == "__main__":
    run_mnl(delta=DELTA, eps=EPS)
    
  
