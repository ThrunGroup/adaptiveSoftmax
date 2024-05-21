import os
import time
import numpy as np
import pandas as pd

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
    FUDGE_TRAIN_SIZE,

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
            train_size = 200,
            exact_pull_best_arm=False  # this could work if variance is small???
        ))
        
def run_mnl(dataset, delta, eps, curr_time=None):
    curr_time = curr_time if curr_time else time.strftime("%H:%M:%S", time.gmtime())
    save_dir = f"{MNL_RESULTS_DIR}/{curr_time}/{dataset}"
    os.makedirs(save_dir, exist_ok=True)

    # run sftm if first time
    save_path = f"{save_dir}/delta{delta}_eps{eps}"
    if not any(os.scandir(save_dir)):
        loading_path = MNIST_FINAL_PATH if dataset == MNIST else EUROSAT_FINAL_PATH
        A = np.load(f"{MNL_WEIGHTS_DIR}/{loading_path}")['data']
        X = np.load(f"{MNL_XS_DIR}/{loading_path}")['data']

        run(
            save_to=f"{save_path}_raw.csv",
            model="CNN",
            dataset=dataset,
            A=A,
            X=X[:NUM_QUERIES],  # default 1000
            multiplicative_error=eps,
            failure_probability=delta,
            noise_bound=None,
            use_true_sftm=False,
            use_tune=True,
            train_size=FUDGE_TRAIN_SIZE,  # default 200
            exact_pull_best_arm=False  
        )

    # plot the results
    data = pd.read_csv(f"{save_path}_raw.csv")
    data = clean_singleton_np_array_columns(data)
    budget, success_rate = get_budget_and_success_rate(data)
    pd.DataFrame([{"budget": budget, "delta accuracy": success_rate}]).to_csv(f"{save_path}_results.csv")

if __name__ == "__main__":
    for dataset in [MNIST, EUROSAT]:
        run_mnl(
            dataset,
            delta=DELTA, 
            eps=EPS,
            curr_time=None,  # pass in specific time here if you don't want to rerun
        )
    
  
