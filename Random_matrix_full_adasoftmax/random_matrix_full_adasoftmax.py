import numpy as np
import matplotlib.pyplot as plt
import torch
from hadamard_transform import hadamard_transform
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptive_softmax'))
from adasoftmax import ada_softmax, approx_sigma
from constants import PROFILE, RETURN_STAGE_BUDGETS


if __name__ == "__main__":
    np.random.seed(42)
    # flag variables
    use_hadamard_transform = False
    verbose = False
    verbose_hadamard = False

    # number of arms(n in the original paper)
    n = 10

    # hyperparameter for adasoftmax
    beta = 1
    epsilon = 0.1
    delta = 0.01
    k = 1

    N_EXPERIMENTS = 10

    # for plotting
    dimension_list = list()
    budget_list = list()

    for d in range(10000, 11000, 1000):
        dimension_list.append(d)
        print("dimension:", d)

        # test statistics aggregate
        error_sum = 0.0
        wrong_approx_num = 0
        total_budget = 0
        if PROFILE:
            total_norm_budget = 0
            total_norm_armid_budget = 0
            alpha_error = np.zeros(2)
            gamma_error = np.zeros(2)
            top_2_T2 = np.zeros(2)
            top_2_T3 = np.zeros(2)

        elif RETURN_STAGE_BUDGETS:
            total_norm_budget = 0
            total_norm_armid_budget = 0


        if use_hadamard_transform:
            dPad = int(2 ** np.ceil(np.log2(d)))
            D = np.diag(np.random.choice([-1, 1], size=dPad))

        for i in range(N_EXPERIMENTS):
            if verbose:
                print(i)
            # data generation in L42-L55
            A = np.random.normal(loc=0, scale=1, size=(n, d))

            # normalize all rows of A to have same l2 norm
            # this is to ensure that i = argmax(A@A[i]), with reasonable "gain"
            A_norm = np.linalg.norm(A, axis=1)
            for j in range(n):
                A[j] = A[j] / (A_norm[j] / 2.4)

            best_index = np.random.choice(10)

            # TODO(@lukehan): normalize the gaussian noise -> scale to desired constant(1e-3, for example)
            #       also, add hyperparameter to control the scale of noise
            gaussian_noise = np.random.normal(loc=0, scale=1e-3, size=(d,))
            x = A[best_index] + gaussian_noise

            mu = A @ x
            z = np.exp(mu - mu.max()) / np.sum(np.exp(mu - mu.max()))

            gain = n * np.sum(np.exp(2 * (mu - mu.max()))) / (np.sum(np.exp(mu - mu.max())) ** 2)
            print(gain)

            if verbose:
                gain = n * np.sum(np.exp(2 * (mu - mu.max()))) / (np.sum(np.exp(mu - mu.max())) ** 2)
                print(gain)

            if use_hadamard_transform:
                # pad A and x to shape (n, dPad) and (dPad,) respectively,
                # where
                Apad = np.pad(A, ((0, 0), (0, dPad - d)), 'constant', constant_values=0)
                xpad = np.pad(x, (0, dPad - d), 'constant', constant_values=0)

                # convert padded A and x to Tensor in order to use pytorch's hadamard transform library
                A_pad_torch = torch.tensor(Apad)
                x_pad_torch = torch.tensor(xpad)

                Aprime = hadamard_transform(A_pad_torch @ D).numpy()
                xprime = hadamard_transform(x_pad_torch @ D).numpy()

                if verbose_hadamard:
                    print("is valid transform", np.allclose(A@x, Aprime@xprime))
                    print("original sigma:", approx_sigma(A, x, d))
                    print("transform sigma:", approx_sigma(Aprime, xprime, dPad))

                best_index_hat, z_hat, bandit_budget = ada_softmax(A=Aprime,
                                                                   x=xprime,
                                                                   epsilon=epsilon,
                                                                   delta=delta,
                                                                   samples_for_sigma=d,
                                                                   beta=beta,
                                                                   k=k,
                                                                   )
            else:
                # Run adaSoftmax with untransformed A and x
                if PROFILE:
                    best_index_hat, z_hat, bandit_budget, profiling_results = ada_softmax(A=A,
                                                                                           x=x,
                                                                                           epsilon=epsilon,
                                                                                           delta=delta,
                                                                                           samples_for_sigma=d,
                                                                                           beta=beta,
                                                                                           k=k,
                                                                                           verbose=False
                                                                                           )
                elif RETURN_STAGE_BUDGETS:
                    best_index_hat, z_hat, bandit_budget, stage_budgets = ada_softmax(A=A,
                                                                                          x=x,
                                                                                          epsilon=epsilon,
                                                                                          delta=delta,
                                                                                          samples_for_sigma=d,
                                                                                          beta=beta,
                                                                                          k=k,
                                                                                          verbose=False
                                                                                          )
                else:
                    best_index_hat, z_hat, bandit_budget = ada_softmax(A=A,
                                                                      x=x,
                                                                      epsilon=epsilon,
                                                                      delta=delta,
                                                                      samples_for_sigma=d,
                                                                      beta=beta,
                                                                      k=k,
                                                                      verbose=False
                                                                      )

            total_budget += bandit_budget
            if PROFILE:
                total_norm_budget += profiling_results["denom budget"]
                total_norm_armid_budget += profiling_results["denom+best-arm budget"]
                top_2_T2 += profiling_results["top-2 T2"]
                top_2_T3 += profiling_results["top-2 T3"]
                print("\n---------------------------------------------------------------")
                print("n(number of arms):", n)
                print("d(dimension):", d)
                print("naive budget:", n*d)
                print("sigma(median among all arms):", profiling_results["sigma"])
                print("denominator estimation budget:", profiling_results["denom budget"])
                print("additional budget spent on best-arm identification:", profiling_results["denom+best-arm budget"] - profiling_results["denom budget"])
                print("additional budget spent on numerator estimation:", bandit_budget - profiling_results["denom+best-arm budget"])
                print("T0:", profiling_results["T0"])
                print("top-2 T2(epsilon^-1):", profiling_results["top-2 T2"])
                print("top-2 T3(epsilon^-2):", profiling_results["top-2 T3"])
                print("top-2 alpha's ratio to true alpha from behind:", profiling_results["alpha_error"])
                print("first order error:", "{:.1e}".format(profiling_results["first_order_error"]))
                print("second order error:", "{:.1e}".format(profiling_results["second_order_error"]))
                print("error on denominator estimation:", "{:.1e}".format(profiling_results["denominator error"]))
                print("error on numerator estimation:", "{:.1e}".format(profiling_results["numerator error"]))
                print("---------------------------------------------------------------\n")

            elif RETURN_STAGE_BUDGETS:
                total_norm_budget += stage_budgets["denom budget"]
                total_norm_armid_budget += stage_budgets["denom+best-arm budget"]

            cur_epsilon = np.abs(z_hat[best_index_hat] - z[best_index_hat]) / z[best_index_hat]

            if cur_epsilon[0] <= epsilon and best_index_hat[0] == np.argmax(z): #ASSUMING K=1
                error_sum += cur_epsilon[0]
            elif best_index_hat[0] == np.argmax(z):
                wrong_approx_num += 1
                error_sum += cur_epsilon[0]
            else:
                wrong_approx_num += 1
                if verbose:
                    print(bandit_budget)
                    print(z)
                    print(z_hat)
                    print(best_index_hat[0], np.argmax(z), cur_epsilon[0])

        imp_delta = wrong_approx_num / N_EXPERIMENTS
        average_budget = total_budget / N_EXPERIMENTS
        imp_epsilon = error_sum / N_EXPERIMENTS

        # TODO: Only for profiling purpose
        if PROFILE:
            average_norm_budget = total_norm_budget / N_EXPERIMENTS
            average_norm_armid_budget = total_norm_armid_budget / N_EXPERIMENTS
            average_top2_T2 = top_2_T2 / N_EXPERIMENTS
            average_top2_T3 = top_2_T3 / N_EXPERIMENTS
            print("average budget for normalization estimation:", average_norm_budget)
            print("average budget until best-arm identification:", average_norm_armid_budget)
            print("average top-2 T2:", average_top2_T2)
            print("average top-2 T3:", average_top2_T3)

        elif RETURN_STAGE_BUDGETS:
            average_norm_budget = total_norm_budget / N_EXPERIMENTS
            average_norm_armid_budget = total_norm_armid_budget / N_EXPERIMENTS


        print("=>average budget:", average_budget)
        print("\n")
        print("=>delta:", imp_delta)
        print("=>average error:", imp_epsilon)
        print('\n\n')

        budget_list.append(average_budget)


    plt.plot(dimension_list, budget_list, "r--.", label="adaptive_softmax")
    plt.plot(dimension_list, n * np.array(dimension_list), "b--.", label="naive")
    plt.legend()
    plt.xlabel("dimension(n_features)")
    plt.ylabel("number of samples taken")
    plt.savefig("sample_complexity_plot.png", bbox_inches="tight")
    plt.yscale("log")
    plt.plot(dimension_list, budget_list, "r--.", label="adaptive_softmax")
    plt.plot(dimension_list, n * np.array(dimension_list), "b--.", label="naive")
    plt.savefig("sample_complexity_log_plot.png", bbox_inches="tight")