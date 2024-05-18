import numpy as np
import matplotlib.pyplot as plt
import statistics
from tqdm import tqdm

from adaptive_softmax.sftm import SFTM
from tests.test_utils import construct_sanity_example, construct_noisy_example, single_run_adasoftmax
from experiments.runner import run
from adaptive_softmax.constants import (
    SCALING_POINTS,
    TEST_EPSILON,
    TEST_DELTA,
    TEST_BETA,
    TEST_TOPK,
    NUM_TRIALS,
)

def scaling_synthetic(n, initial_d, is_noisy=False):
    dimensions = []
    budgets = []
    naive_budgets = []
    deltas = []
    variances = []

    for point in tqdm(range(1, SCALING_POINTS)):
        point_budgets = []
        point_errors = []
        curr_d = initial_d * (10 * point)
        for trial in range(NUM_TRIALS):
            np.random.seed(trial)
            if is_noisy:
                A, x = construct_noisy_example(n, curr_d)
                noise_bound = 1/curr_d
            else:
                A, x = construct_sanity_example(n, curr_d)
                noise_bound = 0

            sftm = SFTM(
                A, 
                multiplicative_error=TEST_EPSILON, 
                failure_probability=TEST_DELTA, 
                temperature=TEST_BETA, 
                atom_importance_sampling=False,
                query_importance_sampling=False,
                randomized_hadamard_transform=False,
                exact_pull_best_arm=False,
                noise_bound=noise_bound,
            )

            in_bounds, error, budget = single_run_adasoftmax(
                sftm=sftm,
                x=x,
                k=TEST_TOPK,
            )
            point_budgets.append(budget)
            point_errors.append(int(not in_bounds))
            
        deltas.append(sum(point_errors) // NUM_TRIALS)
        variances.append(statistics.stdev(point_budgets) / len(point_budgets))
        budgets.append(sum(point_budgets) // NUM_TRIALS)
        dimensions.append(curr_d)
        naive_budgets.append(n * curr_d)

    return dimensions, naive_budgets, budgets, variances, deltas


def plot_scaling(dimensions, naive_bdgets, budgets, variances, deltas):
    plt.figure(figsize=(10, 6))
    
    # Plotting budgets vs. dimensions with error bars
    plt.errorbar(dimensions, budgets, yerr=variances, fmt='o-', color='blue', label='Budgets', capsize=5)
    
    # # Plotting naive budgets vs. dimensions
    plt.plot(dimensions, naive_bdgets, 's-', color='red', label='Naive Budgets')
    
    # Adding labels and title
    plt.xlabel('Dimension (d)')
    plt.ylabel('Budget')
    
    # Updating the title to include variance
    plt.title(f'Budget vs Dimension\nAvg Deltas = {sum(deltas)//len(deltas)}')
    
    # Setting the x-axis to a logarithmic scale
    plt.yscale('log')
    
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # dimensions, naive_bdgets, budgets, variances, deltas = scaling_synthetic(n=100, initial_d=1000, is_noisy=False)
    # plot_scaling(dimensions, naive_bdgets, budgets, variances, deltas)
    # print(budgets)

    dimensions, naive_bdgets, budgets, variances, deltas = scaling_synthetic(n=100, initial_d=1000, is_noisy=True)
    print(budgets)
    plot_scaling(dimensions, naive_bdgets, budgets, variances, deltas)
