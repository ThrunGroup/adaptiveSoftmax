import numpy as np
import torch
from typing import Tuple

from adasoftmax import (
    ada_softmax,
    estimate_mu_hat,
    find_topk_arms,
    approx_sigma,
)


def normalization_estimation_test(
    true_mu: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    sigma: float,
    beta: float,
    delta: float,
    epsilon: float,
) -> Tuple[float, bool, int]:
    """
    Tests the correctness of normalization constant(referred as S below) estimation.
    Specifically, this function tests if the approximation from the normalization constant estimation algorithm
    is within the multiplicative error of epsilon / 2 (Refer to the paper for this number).
    Note that the test on delta(error probability) by testing on multiple instances
    would be done outside the function.

    :param true_mu: ground truth mu
    :param A: matrix A in the paper(A@x == true_mu)
    :param x: vector x in the paper(A@x == true_mu)
    :param sigma: sigma in the paper(std of the arm pull)
    :return: multiplicative error of the estimation, indicator for the correctness, budget for normalization estimation
    """
    # Calculate ground truth for normalization constant
    true_S = np.sum(np.exp(beta * true_mu))

    # Estimate the normalization constant using the algorithm
    mu_hat_norm, budget_vec_norm = estimate_mu_hat(
        atoms=A,
        query=x,
        epsilon=epsilon / 2,
        delta=delta / 3,
        sigma=sigma,
        beta=beta,
        dist=np.ones(len(A[0])) / sum(np.ones(len(A[0]))),
    )
    S_estimate = np.sum(np.exp(beta * mu_hat_norm))

    S_error = np.abs(S_estimate - true_S) / true_S
    is_correct = (
        S_error <= epsilon / 2
    )  # Refer to algorithm 2, Line 3 in original paper for epsilon bound
    normalization_estimation_budget = np.sum(budget_vec_norm).item()

    return S_error, is_correct, normalization_estimation_budget


def topk_identification_test(
    true_mu: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    n: int,
    k: int,
    delta: float,
    sigma: float,
) -> Tuple[bool, int]:
    """
    Tests the correctness of normalization constant(referred as S below) estimation.
    Specifically, this function tests if the topk-identification successfully identifies the best indices.
    Note that the test on delta(error probability) by testing on multiple instances
    would be done outside of this function.

    :param true_mu: ground truth
    :param A: matrix A in the paper(A@x == true_mu)
    :param x: vector x in the paper(A@x == true_mu)
    :param sigma: sigma in the paper(std of the arm pull)
    :return: indicator for identifying the best indices correctly, and the budget for topk-identification
    """
    # Find true top-k indices
    true_topk_indices_torch = torch.sort(
        torch.topk(torch.from_numpy(true_mu), k).indices
    ).values
    true_topk_indices = true_topk_indices_torch.numpy()

    # Calculate mu estimate for find-topk
    # This vector would be constructed by estimate_softmax_normalization in ada_softmax
    topk_start_mu_hat = np.array([A[i][0] * x[0] for i in range(n)])
    topk_start_budget = np.ones(n).astype(np.int64)

    # Estimate the best indices using the algorithm
    best_indices_topk, _, budget_vec_topk = find_topk_arms(
        atoms=A,
        query=x,
        sigma=sigma,
        delta=delta / 3,
        mu_hat=topk_start_mu_hat,
        d_used=topk_start_budget,
        k=k,
        dist=np.ones(len(A[0])) / sum(np.ones(len(A[0]))),
    )

    # Test results
    estimated_right_topk = np.allclose(np.sort(best_indices_topk), true_topk_indices)
    topk_identification_budget = np.sum(budget_vec_topk).item()

    # Print ground truth and estimate if the algorithm is not correct
    if not estimated_right_topk:
        print(true_topk_indices, best_indices_topk)

    return estimated_right_topk, topk_identification_budget


def ada_softmax_test(
    true_mu: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    k: int,
    d: int,
    beta: float,
    delta: float,
    epsilon: float,
) -> Tuple[float, bool, bool, int]:
    """
    Tests the correctness of the adaSoftmax algorithm.
    Specifically, it tests if the adaSoftmax successfully identifies the best indices,
    and estimates the softmax value for those indices within the multiplicative error of epsilon.
    Note that the test on delta(error probability) is done outside the function.

    :param true_mu: ground truth mu
    :param A: matrix A in the paper(A@x == true_mu)
    :param x: vector x in the paper(A@x == true_mu)
    :return: multiplicative error of the estimation,
             indicator for identifying the best indices,
             indicator for estimating within multiplicative error bound,
             budget for adaSoftmax
    """
    # Calculate ground truth for softmax result
    S = np.sum(np.exp(beta * true_mu))
    z = np.exp(beta * true_mu) / S

    # Find true top-k indices
    true_topk_indices_torch = torch.sort(
        torch.topk(torch.from_numpy(true_mu), k).indices
    ).values
    true_topk_indices = true_topk_indices_torch.numpy()

    # Identify the best indices, estimate the softmax value for best indices
    best_indices_hat, z_hat, adasoftmax_budget = ada_softmax(
        A=A,
        x=x,
        epsilon=epsilon,
        delta=delta,
        samples_for_sigma=d,
        beta=beta,
        k=k,
        verbose=False,
    )

    best_indices_hat = np.sort(best_indices_hat)

    # Test results
    estimated_right_indices = np.allclose(best_indices_hat, true_topk_indices)
    z_error = np.abs(z_hat[best_indices_hat] - z[true_topk_indices])
    empirical_epsilon = np.max(z_error / z[true_topk_indices])
    within_error_bound = empirical_epsilon <= epsilon

    # Print ground truth, estimate, and related values if the algorithm is not correct
    if not estimated_right_indices:
        print(true_topk_indices, best_indices_hat)
    elif not within_error_bound:
        print("epsilon:", empirical_epsilon)
        print(best_indices_hat, true_topk_indices)
        print("approx, truth:", z_hat[best_indices_hat], z[true_topk_indices])
        print("z_hat:", z_hat)
        print("z:", z)
        print(A @ x)

    return (
        empirical_epsilon,
        estimated_right_indices,
        within_error_bound,
        adasoftmax_budget,
    )


def main():
    NUM_TESTS = 100
    verbose = True

    # matrix, vector size
    n = 10
    d = int(3e4)

    # adaptive algorithm hyperparameters
    beta = 1
    epsilon = 0.1
    delta = 0.01
    num_sigma_samples = d
    k = 1

    np.random.seed(42)

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
        true_mu = np.random.uniform(1, 100, size=(n,))
        true_mu /= 10

        # construct A and x that satisfies A@x = true_mu
        x = np.random.uniform(low=0.94, high=1, size=d)
        Z = np.random.normal(loc=0, scale=1 / d, size=(n, d))
        A = np.outer(true_mu, x) / np.sum(x**2) + Z
        A = A - np.outer(A @ x - true_mu, np.ones(d) / np.sum(x))

        if verbose:
            print(f"running {i+1}th test")

        # calculate true sigma
        sigma, _dist = approx_sigma(A, x, num_sigma_samples)

        # normalization constant estimation test
        S_error, S_within_bound, normalization_estimation_budget = (
            normalization_estimation_test(
                true_mu=true_mu,
                A=A,
                x=x,
                sigma=sigma,
                beta=beta,
                delta=delta,
                epsilon=epsilon,
            )
        )

        # Aggregate test result
        total_S_error += S_error
        if not S_within_bound:
            wrong_S_estimate_numbers += 1
        total_estimate_normalization_budget += normalization_estimation_budget

        # top-k indices identification test
        estimated_right_topk, topk_budget = topk_identification_test(
            true_mu=true_mu,
            A=A,
            x=x,
            n=n,
            k=k,
            delta=delta,
            sigma=sigma,
        )

        if not estimated_right_topk:
            wrong_topk_numbers += 1
        else:
            total_topk_budget += topk_budget

        # softmax estimation test
        empirical_epsilon, estimated_right_indices, within_error_bound, budget = (
            ada_softmax_test(
                true_mu=true_mu,
                A=A,
                x=x,
                k=k,
                d=d,
                beta=beta,
                delta=delta,
                epsilon=epsilon,
            )
        )

        # print("budget: ", budget)

        # Aggregate test results
        if estimated_right_indices and within_error_bound:
            total_softmax_error += empirical_epsilon
            total_softmax_budget += budget
        else:
            wrong_softmax_estimate_numbers += 1

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
    print(
        "empirical delta for softmax estimation:",
        wrong_softmax_estimate_numbers / NUM_TESTS,
    )
    print("budget for softmax estimation:", total_softmax_budget / NUM_TESTS)


if __name__ == "__main__":
    main()
