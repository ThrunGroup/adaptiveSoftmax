import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from experiments.runner import run
from tests.test_utils import construct_sanity_example, construct_noisy_example
from experiments.plotter import plot_scaling, get_scaling_param, get_budget_and_success_rate
from adaptive_softmax.constants import (
    SCALING_POINTS,
    NUM_TRIALS,
    SYNTHETIC_RESULT_DIR,
    TOY_GAUSSAIN,
    REGULAR_GUASSIAN,
)

def get_run_results(run_data: pd.DataFrame):
    per_arm_budget = np.mean(run_data['budget_total'] / (run_data['n']))
    return per_arm_budget

def scaling_synthetic(n, dataset):
    avg_per_arm_budgets = []
    std_errs = []

    d = (10 ** np.linspace(3, 6, 10)).astype(int)

    for curr_d in tqdm(d):
        per_arm_budgets = []
    
        for trial in range(NUM_TRIALS):
            np.random.seed(trial)
            
            # NOTE: need to construct A each time too. Otherwise, exact search
            if dataset == REGULAR_GUASSIAN:
                A, x = construct_noisy_example(n, curr_d) 
                noise_bound = 1
            else: 
                A, x = construct_sanity_example(n, curr_d)
                noise_bound = 1/10

            run_data = run(
                save_to=None,
                model="synthetic",
                dataset=dataset,
                A=A,
                X=np.array(x, ndmin=2),  # this is for compabitility with runner.py
                multiplicative_error=0.3,
                failure_probability=0.1,
                noise_bound = noise_bound,
                use_true_sftm = True,
                use_tune = False,
                train_size = 1,
                quiet=True,
            )

            per_arm_budget = get_run_results(run_data)
            per_arm_budgets.append(per_arm_budget)

        avg_per_arm_budgets.append(np.mean(per_arm_budgets))
        std_errs.append(np.std(per_arm_budgets) / np.sqrt(NUM_TRIALS))

    run_data = {
        'd': d,
        'per_arm_budgets': avg_per_arm_budgets,
        'stderr': std_errs,
    }

    return run_data

        
def run_and_plot_synthetic(n):
    paths = []
    for dataset in [TOY_GAUSSAIN, REGULAR_GUASSIAN]:
        save_dir = f"{SYNTHETIC_RESULT_DIR}/{dataset}/n{n}"
        os.makedirs(save_dir, exist_ok=True)
        run_data = scaling_synthetic(n=n, dataset=dataset)
        path = plot_scaling(run_data, save_dir)
        paths.append(path)
    return paths


if __name__ == "__main__":
    run_and_plot_synthetic(n=100)
    
  
