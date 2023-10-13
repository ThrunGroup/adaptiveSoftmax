import numpy as np
import matplotlib.pyplot as plt
import torch
from hadamard_transform import hadamard_transform
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptive_softmax'))
from adasoftmax import ada_softmax, approx_sigma


if __name__ == "__main__":
    np.random.seed(777)
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

    for d in range(40000, 50000, 10000):
        dimension_list.append(d)

        if verbose:
            print("dimension:", d)

        # test statistics aggregate
        error_sum = 0.0
        wrong_approx_num = 0
        total_budget = 0

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

        print("=>delta:", imp_delta)
        print("=>average budget:", average_budget)
        print("=>average error:", imp_epsilon)

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