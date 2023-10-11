import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import torch
from hadamard_transform import randomized_hadamard_transform, hadamard_transform
np.random.seed(777)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptive_softmax'))
from adasoftmax import ada_softmax


if __name__ == "__main__":
    # flag variables
    use_hadamard_transform = False
    verbose = True

    # number of arms(n in the original paper)
    n = 10

    # hyperparameter for adasoftmax
    beta = 1
    epsilon = 0.1
    delta = 0.01
    k = 1

    N_EXPERIMENTS = 100

    # for plotting
    dimension_list = list()
    budget_list = list()

    for d in range(10000, 110000, 10000):
        dimension_list.append(d)
        print("dimension:", d)

        # test statistics aggregate
        error_sum = 0.0
        wrong_approx_num = 0
        total_budget = 0

        for i in range(N_EXPERIMENTS):

            A = np.random.normal(loc=0, scale=0.1, size=(n, d))

            # normalize all rows of A to have same l2 norm
            # this is to ensure that i = argmax(A@A[i]), with reasonable "gain"
            A_norm = np.linalg.norm(A, axis=1)
            for j in range(n):
                A[j] = A[j] / (A_norm[j] / 2.4)


            x = A[i] #Need to change
            mu = A @ x
            mu -= np.max(mu)
            z = np.exp(mu) / np.sum(np.exp(mu))

            gain = n * np.sum(np.exp(2 * (mu - np.max(mu)))) / (np.sum(np.exp(mu - np.max(mu)))**2)
            #print("gain:", gain)
            """
            transform_mu = 
            transform_gain = 
            """

            best_index_hat, z_hat, bandit_budget = ada_softmax(A=A,
                                                               x=x,
                                                               epsilon=epsilon,
                                                               delta=delta,
                                                               samples_for_sigma=d,
                                                               beta=beta,
                                                               k=k,
                                                               )

            #print("best_index:", best_index_hat)


            total_budget += bandit_budget

            cur_epsilon = np.abs(z_hat[best_index_hat] - z[best_index_hat]) / z[best_index_hat]
            #print(z_hat[best_index_hat], z[best_index_hat], np.max(z))

            if cur_epsilon[0] > 1e-2:
                print(cur_epsilon)

            if cur_epsilon[0] <= epsilon and best_index_hat[0] == np.argmax(z): #ASSUMING K=1
                error_sum += cur_epsilon[0]
            elif best_index_hat[0] == np.argmax(z):
                wrong_approx_num += 1
                error_sum += cur_epsilon[0]
            else:
                wrong_approx_num += 1
                print(bandit_budget)
                print(z)
                print(z_hat)
                print(best_index_hat[0], np.argmax(z), cur_epsilon[0])

        imp_delta = wrong_approx_num / N_EXPERIMENTS
        average_budget = total_budget / N_EXPERIMENTS
        imp_epsilon = error_sum / N_EXPERIMENTS

        print("=>delta:", imp_delta)
        print("=>average budget:", average_budget)
        print("=>average error:", imp_epsilon)
        #print("=>wrong_approx_num:", wrong_approx_num)

        budget_list.append(average_budget)


    plt.plot(dimension_list, budget_list, "r--.", label="adaptive_softmax")
    plt.plot(dimension_list, n * np.array(dimension_list), "b--.", label="naive")
    plt.legend()
    plt.xlabel("dimension(n_features)")
    plt.ylabel("number of samples taken")
    plt.savefig("sample_complexity_plot.svg", bbox_inches="tight")
    plt.yscale("log")
    plt.plot(dimension_list, budget_list, "r--.", label="adaptive_softmax")
    plt.plot(dimension_list, n * np.array(dimension_list), "b--.", label="naive")
    plt.savefig("sample_complexity_log_plot.svg", bbox_inches="tight")