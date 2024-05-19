import os
import numpy as np
import pandas as pd

from tests.test_utils import construct_sanity_example, construct_noisy_example
from experiments.plotter import clean_singleton_np_array_columns, get_budget_and_success_rate, plot_scaling
from experiments.runner import run
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
            A, x = construct_noisy_example(n, curr_d)
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

        # this is for all d
        dimensions = []
        budgets = []
        naive_budgets = []
        success_rates = []

        for file in sorted(os.listdir(path_dir)):
            data = pd.read_csv(os.path.join(path_dir, file))
            data = clean_singleton_np_array_columns(data)

            dimensions.append(int(np.mean(data['d'])))
            budgets.append(int(np.mean(data['budget_total'])))
            naive_budgets.append(int(np.mean(data['d'] * data['n'])))
            success_rates.append(get_budget_and_success_rate(data)[1])

        save_to = f"experiments/synthetic_results/plots"
        os.makedirs(save_to, exist_ok=True)
        plot_scaling(dimensions, naive_budgets, budgets, success_rates, f"{save_to}_noisy_is_{is_noisy}_n={n}")

if __name__ == "__main__":
    run_synthetic(n=100, init_d=1000)
    
  
