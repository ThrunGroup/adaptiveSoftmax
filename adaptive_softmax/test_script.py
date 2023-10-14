import numpy as np
from adasoftmax import ada_softmax, estimate_mu_hat, find_topk_arms, approx_sigma
import torch

if __name__ == "__main__":
    np.random.seed(42)

    NUM_TESTS = 1000
    verbose = True

    # constant for adjusting gain.
    c = 5
    # matrix, vector size
    n = 10
    d = int(3e+4)

    # adaptive algorithm hyperparameters
    beta = 1
    epsilon = 0.1
    delta = 0.01
    num_sigma_samples = d
    k = 1

    # test result aggregates for softmax normalization estimation
    total_S_error = 0.0
    wrong_S_estimate_numbers = 0
    total_estimate_normalization_budget = 0

    # test result aggregates for top-k indices identification
    wrong_topk_numbers = 0
    total_topk_budget = 0

    # test result aggregates for softmax estimation
    wrong_softmax_estimate_numbers = 0
    total_softmax_error = 0.0
    total_softmax_budget = 0

    for i in range(NUM_TESTS):
        # generate ground truth mu randomly
        """
        true_mu = np.ones(n)
        true_mu = np.random.uniform(1, 100, size=(n,))
        true_mu /= 10
        """
        true_mu = np.array([2.84174176, 0.21693943, 4.94935448, 8.54784365, 6.60296774, 0.25694716,
                            0.1692839, 8.39057803, 8.22458089, 8.52506058])

        # find true top-k indices
        true_topk_indices_torch = torch.sort(torch.topk(torch.from_numpy(true_mu), k).indices).values
        true_topk_indices = true_topk_indices_torch.numpy()

        # construct A and x that satisfies A@x = true_mu
        x = np.random.uniform(low=0.94, high=1, size=d)
        Z = np.random.normal(loc=0, scale=1 / d, size=(n, d))
        A = np.outer(true_mu, x) / np.sum(x ** 2) + Z
        A = A - np.outer(A @ x - true_mu, np.ones(d) / np.sum(x))

        # calculate ground truth for S and z
        S = np.sum(np.exp(beta * true_mu))
        z = np.exp(beta * true_mu) / S

        if verbose:
            print(f"running {i+1}th test")

        # calculate true sigma
        sigma = approx_sigma(A, x, num_sigma_samples)


        # normalization constant estimation test
        mu_hat_norm, budget_vec_norm, profiling_results = estimate_mu_hat(atoms=A,
                                                       query=x,
                                                       epsilon=epsilon/2,
                                                       delta=delta/3,
                                                       sigma=sigma,
                                                       beta=beta,
                                                       )
        S_hat = np.sum(np.exp(beta * mu_hat_norm))
        # print(np.exp(mu_hat_norm))
        # print(np.exp(beta * true_mu))
        S_error = np.abs(S_hat - S) / S
        total_S_error += S_error
        # Refer to algorithm 2, Line 3 in original paper for epsilon bound on normalization constant estimation
        if S_error > epsilon/2:
            wrong_S_estimate_numbers += 1
        total_estimate_normalization_budget += np.sum(budget_vec_norm).item()

        # calculate mu estimate for find-topk
        # This vector would be constructed by estimate_softmax_normalization in ada_softmax
        topk_start_mu_hat = np.array([A[i][0]*x[0] for i in range(n)])
        topk_start_budget = np.ones(n).astype(np.int64)

        # top-k indices identification test
        best_indices_topk, _, budget_vec_topk = find_topk_arms(atoms=A,
                                                               query=x,
                                                               sigma=sigma,
                                                               delta=delta/3,
                                                               mu_approx=topk_start_mu_hat,
                                                               d_used=topk_start_budget,
                                                               k=k,
                                                               )
        estimated_right_topk = np.allclose(np.sort(best_indices_topk), true_topk_indices)
        if not estimated_right_topk:
            wrong_topk_numbers += 1
            print(best_indices_topk, true_topk_indices, true_mu)
        else:
            total_topk_budget += np.sum(budget_vec_topk).item()

        # softmax estimation test
        best_indices_hat, z_hat, adasoftmax_budget = ada_softmax(A=A,
                                                                 x=x,
                                                                 epsilon=epsilon,
                                                                 delta=delta,
                                                                 samples_for_sigma=d,
                                                                 beta=beta,
                                                                 k=k,
                                                                 )

        best_indices_hat = np.sort(best_indices_hat)

        estimated_right_indices = np.allclose(best_indices_hat, true_topk_indices)
        if not estimated_right_indices:
            wrong_softmax_estimate_numbers += 1
        else:
            z_error = np.abs(z_hat[best_indices_hat] - z[true_topk_indices])
            empirical_epsilon = np.max(z_error / z[true_topk_indices])
            if empirical_epsilon > epsilon:
                print("epsilon:", empirical_epsilon)
                print(best_indices_hat, true_topk_indices)
                print(z_hat[best_indices_hat], z[true_topk_indices])
                print(A@x)
                wrong_softmax_estimate_numbers += 1
            else:
                total_softmax_error += empirical_epsilon
                total_softmax_budget += adasoftmax_budget

    average_S_error = total_S_error / NUM_TESTS
    average_softmax_error = total_softmax_error / NUM_TESTS

    print("----------------------------------------------------")
    print("desired S estimation error:", epsilon / 2)
    print("average empirical S estimation error:", average_S_error)
    print("desired delta for S estimation:", delta / 3)
    print("empirical delta for S estimation:", wrong_S_estimate_numbers / NUM_TESTS)
    print("budget for S estimation:", total_estimate_normalization_budget / NUM_TESTS)
    print("----------------------------------------------------")
    print("desired delta for topk identification:", delta / 3)
    print("empirical delta for topk identification:", wrong_topk_numbers / NUM_TESTS)
    print("budget for topk identification:", total_topk_budget / NUM_TESTS)
    print("----------------------------------------------------")
    print("desired softmax estimation error:", epsilon)
    print("average empirical softmax estimation error:", average_softmax_error)
    print("desired delta for softmax estimation:", delta)
    print("empirical delta for softmax estimation:", wrong_softmax_estimate_numbers / NUM_TESTS)
    print("budget for softmax estimation:", total_softmax_budget / NUM_TESTS)