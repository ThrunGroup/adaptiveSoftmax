import os
import numpy as np
import pandas as pd

from experiments.runner import run
from tests.test_utils import construct_sanity_example, construct_noisy_example
from experiments.plotter import plot_scaling, get_scaling_param
from adaptive_softmax.constants import (
    SCALING_POINTS,
    NUM_TRIALS,
)

def scaling_synthetic(n, initial_d, is_noisy, path_dir):
    for point in range(1, SCALING_POINTS):
        curr_d = initial_d * (10 * point)
    
        for trial in range(NUM_TRIALS):
            np.random.seed(trial)
            
            # NOTE: need to construct A each time too. Otherwise, exact search
            A, x = construct_noisy_example(n, curr_d) if is_noisy else construct_sanity_example(n, curr_d)
            model = "scaling synthetic"
            dataset = f"noisy is {is_noisy}"
            path = f"{path_dir}/d={curr_d}"

            print(run(
                save_to=path,
                model=model,
                dataset=dataset,
                A=A,
                X=np.array(x, ndmin=2),  # this is for compabitility with runner.py
                multiplicative_error = 0.3,
                failure_probability = 0.01,
                noise_bound = None,
                use_true_sftm = True,
                use_tune = False,
                train_size = 1,
            ))
        
def run_synthetic(n, init_d):
    for is_noisy in [True, False]:
        path_dir = f"experiments/synthetic_results/noisy_is_{is_noisy}_n={n}"
        os.makedirs(path_dir, exist_ok=True)

        # only run if files don't exist
        if not any(os.scandir(path_dir)):
            scaling_synthetic(n=n, initial_d=init_d, is_noisy=is_noisy, path_dir=path_dir)
        dimensions, budgets, naive_budgets, success_rates = get_scaling_param(path_dir)

        save_to = f"experiments/synthetic_results/plots"
        os.makedirs(save_to, exist_ok=True)
        plot_scaling(dimensions, naive_budgets, budgets, success_rates, f"{save_to}/noisy_is_{is_noisy}_n={n}")

if __name__ == "__main__":
    run_synthetic(n=100, init_d=1000)
    
  
