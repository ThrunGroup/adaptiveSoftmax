import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Any

dir_path = os.path.dirname(os.path.abspath(__file__))
debug_path = os.path.join(dir_path, "debug")
log_file_path = os.path.join(dir_path, "debug", "log.txt")

from .constants import (
    DEBUG,
    DEV_BY,
    DEV_RATIO,
    NUM_BINS,
    IMPORTANCE,
)

np.random.seed(42)


def approx_sigma(
    A: np.ndarray,
    x: np.ndarray,
    num_samples: Any = None,
    importance: bool = IMPORTANCE,
) -> float:
    """
    Function to approximate sigma.
    Currently, We return the "median" of the std for the arm pulls across all arms.

    :param A: Matrix A in the original paper
    :param x: Vector x in the original paper
    :param num_samples: number of samples to use for approximating sigma

    :returns: the sigma approximation
    """
    n, d = A.shape
    dist = np.ones(d) / d  # initially uniform

    # default, get true sigma
    if num_samples is None:
        num_samples = d

    if importance:
        print("\nIMPORTANCE SAMPLING...")
        factor = 0.1
        importance = np.abs(x) / np.sum(np.abs(x))
        dist = (1 - factor) * importance + factor * dist
        dist = dist / dist.sum()  # for numerical stability

    coordinates = np.random.choice(d, p=dist, size=num_samples, replace=False)
    elmul = A[:, coordinates] * x[coordinates] / dist[coordinates]
    sigma = np.mean(np.std(elmul, axis=1))  # empirical variance (maybe use median?)
    print("sigma is ", sigma)

    if DEBUG:
        with open(log_file_path, "a") as f:
            f.write(f"sigma: {sigma}\n")

        # get fraction of deviations that devitate by DEV_BY std (per arms)
        mus = np.mean(elmul, axis=1).reshape(-1, 1)
        devs = np.abs(elmul - mus) / sigma.reshape(-1, 1)
        num_devs = np.sum(devs > DEV_BY, axis=0)
        fraction_per_arms = num_devs / n

        # plot histogram
        bin_edges = np.linspace(0.0, 1.0, NUM_BINS + 1)
        _, bins, _ = plt.hist(fraction_per_arms, bins=bin_edges, edgecolor="black")
        threshold_x = bins[int(DEV_RATIO * NUM_BINS)]
        num_outliers = np.nonzero(fraction_per_arms > DEV_RATIO)[0]

        plt.axvline(x=threshold_x, color="red", linestyle="dashed")
        plt.xlabel(f"fraction")
        plt.ylabel(f"column frequency")
        plt.title(f"columns with fraction of arms greater than {DEV_BY} std")
        plt.text(0.95, 0.95, f"ratio of outliers: {len(num_outliers)/n:.2f}")
        plt.savefig(os.path.join(debug_path, "variance_of_columns.png"))
        plt.close()

    return sigma, dist


def get_importance_errors(
    mu: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
    beta: float,
) -> Tuple[float, float]:
    norm_mu = mu - mu.max()

    true_alpha = np.exp(beta * norm_mu)
    true_alpha = true_alpha / np.sum(true_alpha)
    alpha = alpha / np.sum(alpha)
    alpha_error = np.mean(alpha / true_alpha)

    true_gamma = np.exp((beta * norm_mu) / 2)
    true_gamma = true_gamma / np.sum(true_gamma)
    gamma = gamma / np.sum(gamma)
    gamma_error = np.mean(gamma / true_gamma)

    if DEBUG:
        with open(log_file_path, "a") as f:
            f.write("(alpha, gamma error): ")
            f.write(f"({alpha_error}, {gamma_error})")
            f.write("\n")

    return alpha_error, gamma_error


def get_fs_errors(
    mu: np.ndarray,
    mu_hat: np.ndarray,
    beta: float,
) -> Tuple[float, float]:
    f_error = np.sum(np.exp(beta * mu_hat) * (beta * (mu - mu_hat)))
    f_error /= np.sum(np.exp(mu))

    s_error = np.sum(np.exp(mu_hat) * (beta**2 * (mu - mu_hat) ** 2))
    s_error /= np.sum(np.exp(mu))
    if DEBUG:
        with open(log_file_path, "a") as f:
            f.write(f"(first order, second order): {f_error, s_error}\n")

    return f_error, s_error


def plot_norm_budgets(
    d: float,
    budget: np.ndarray,
    a_error: np.ndarray,
    g_error: np.ndarray,
    f_error: np.ndarray,
    s_error: np.ndarray,
):
    text = f"alpha err: {a_error:.3f}\n"
    text += f"gamma err: {g_error:.3f}\n"
    text += f"fo err: {f_error:.3f}\n"
    text += f"so error: {s_error:.3f}\n"

    bin_edges = np.linspace(0.0, 1.0, NUM_BINS + 1)
    plt.hist(budget / d, bins=bin_edges, edgecolor="black")

    plt.xlabel("ratio of d")
    plt.ylabel("number of arms")
    plt.title("arm pulls for adaptive sampling")
    plt.text(0.95, 0.95, text)

    plt.savefig(os.path.join(debug_path, "normalization_budget.png"))
    plt.close()


def compare_true_arms(
    mu: np.ndarray,
    best_arms: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    best_arms.sort()
    true_best_arms = np.argsort(mu)[-len(best_arms) :]
    true_best_arms.sort()
    diffs = mu[best_arms] - mu[true_best_arms]

    with open(log_file_path, "a") as f:
        f.write(f"algo arms <-> true arms: {best_arms} <-> {true_best_arms}\n")
        f.write(f"difference in mu for these arms: {diffs}\n")

    return true_best_arms, diffs
