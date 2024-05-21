import os
import time
import numpy as np

from experiments.runner import run
from tests.test_utils import construct_sanity_example, construct_noisy_example
from experiments.plotter import plot_scaling, get_scaling_param
from adaptive_softmax.constants import (
    SCALING_POINTS,
    NUM_TRIALS,
    SYNTHETIC_RESULT_DIR,
    TOY_GAUSSAIN,
    REGULAR_GUASSIAN,
)

def scaling_synthetic(n, initial_d, dataset, path_dir):
    for point in range(1, SCALING_POINTS):
        curr_d = initial_d * (10 * point)
    
        for trial in range(NUM_TRIALS):
            np.random.seed(trial)
            
            # NOTE: need to construct A each time too. Otherwise, exact search
            if dataset == REGULAR_GUASSIAN:
                A, x = construct_noisy_example(n, curr_d) 
                noise_bound = 1
            else: 
                A, x = construct_sanity_example(n, curr_d)
                noise_bound = 1

            model = "scaling synthetic"
            path = f"{path_dir}/d={curr_d}"

            print(run(
                save_to=path,
                model=model,
                dataset=dataset,
                A=A,
                X=np.array(x, ndmin=2),  # this is for compabitility with runner.py
                multiplicative_error=0.3,
                failure_probability=0.1,
                noise_bound = noise_bound,
                use_true_sftm = True,
                use_tune = False,
                train_size = 1,
            ))
        
def run_synthetic(n, init_d, curr_time=None):
    curr_time = curr_time if curr_time else time.strftime("%H:%M:%S", time.gmtime())
    for dataset in [TOY_GAUSSAIN, REGULAR_GUASSIAN]:
        save_dir = f"{SYNTHETIC_RESULT_DIR}/{curr_time}/{dataset}/n{n}_init_d{init_d}"
        os.makedirs(save_dir, exist_ok=True)

        if not any(os.scandir(save_dir)):
            scaling_synthetic(n=n, initial_d=init_d, dataset=dataset, path_dir=save_dir)

        dimensions, budgets, naive_budgets, stds, percentages, success_rates = get_scaling_param(save_dir)
        plot_scaling(dimensions, naive_budgets, budgets, stds, percentages, success_rates, save_dir)


if __name__ == "__main__":
    run_synthetic(
        n=100, 
        init_d=1000,
        curr_time = None,
        #curr_time="03:39:18", # pass in specific time here if you don't want to rerun
    )
    
  
