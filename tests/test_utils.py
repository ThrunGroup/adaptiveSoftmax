import numpy as np
from typing import Tuple

from mnl.mnl_utils import load_A_and_xs
from adaptive_softmax.sftm import SFTM
from adaptive_softmax.constants import (
    EPSILON_SCALE,
    DELTA_SCALE,
    TEST_SAMPLES_FOR_SIGMA,
    TEST_MU_LOWER,
    TEST_MU_UPPER,
    TEST_SEED,
)


def construct_sanity_example(
    n: int,
    d: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This is the simplest example for which the adaSoftmax algorithm should always work.
    TLDR; A @ x will be a one-hot vector with nonzero elemement = 1

    :returns: A, x
    """
    A = np.zeros((n, d))
    A[0] = 1 / d
    x = np.ones(d)

    return A, x

def construct_random_example(
    n: int,
    d: int,    
    mu: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly construct A, x to get A @ x = mu. 
    If mu isn't given, create random mu. 

    :returns: A, x
    """
    gen = np.random.default_rng(TEST_SEED)

    # TODO: better way to construct this??
    if mu is None:
        mu = gen.uniform(TEST_MU_LOWER, TEST_MU_UPPER, size=(n,))

    x = gen.uniform(low=0.94, high=1, size=d)  
    Z = gen.normal(loc=0, scale=1 / d, size=(n, d))
    A = np.outer(mu, x) / np.sum(x**2) + Z
    A = A - np.outer(A @ x - mu, np.ones(d) / np.sum(x))

    return A, x

def construct_high_variance_example(
        n: int,
        d: int,
        n_peaks: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random gaussian A with high variance added to some columns and a mostly all ones x
    with higher magnitude at the same columns.
    :param n: number of atoms
    :param d: dimension of query
    :param n_peaks: number of high variance columns 
    :returns: A, x
    """
    gen = np.random.default_rng(TEST_SEED)

    x = np.ones(d)
    A = gen.normal(size=(n, d))
    A[:, :n_peaks] = 10 * gen.choice([-1, 1], size=(n, n_peaks))
    x[:n_peaks] = 10

    return A, x


def single_run_normalization(
    A: np.ndarray,
    x: np.ndarray,
    sigma: float,
    beta: float,
    delta: float,
    epsilon: float,
) -> Tuple[bool, int]:
    """
    Single run of estimate_mu_hat to find normalization constant S in paper (Algo 1, line 3).
    NOTE: the functions names are different from the paper. 
    
    :returns: correctness of S, error, total budget
    """
    true_mu = A @ x
    uniform = np.ones(len(x)) / len(x)
    true_s = np.sum(np.exp(beta * true_mu))

    sftm = SFTM(
        A,
        noise_bound=sigma ** 2,
        failure_probability=delta / DELTA_SCALE,
        multiplicative_error=epsilon / EPSILON_SCALE,
        temperature=beta,
    )

    _, _, s_hat = sftm.adaptive_softmax(x)
    budget = sftm.bandits.it

    error = np.abs(s_hat - true_s) / true_s
    in_bounds = (error <= epsilon / EPSILON_SCALE)  
    total_budget = np.sum(budget).item()

    return in_bounds, error, total_budget, 


def single_run_topk(
    k: int,
    A: np.ndarray,
    x: np.ndarray,
    delta: float,
    sigma: float,
    starting_mu: np.ndarray = None,
    starting_budget: np.ndarray = None,
) -> Tuple[bool, int]:
    """
    Single run of find_topk_arms to find best arm i* in paper (Algo 1, line 5).
    :returns: correctness of i*, total_budget
    """
    # these are arrays
    true_mu = A @ x
    uniform = np.ones(len(x)) / len(x)
    true_topk = np.sort(np.argpartition(true_mu, -k)[-k:])

    # in full algorithm, these values are obtained from estimate_mu_hat
    if starting_mu is None:
        starting_mu = np.zeros(A.shape[0])   
    if starting_budget is None:
        starting_budget = np.ones(A.shape[0]).astype(np.int64)  # zero arms pulled

    sftm = SFTM(
        A,
        noise_bound=sigma ** 2,
        failure_probability=delta / DELTA_SCALE,
    )
    
    topk_hat, _, _ = sftm.adaptive_softmax(x)
    budgets = sftm.bandits.it

    is_correct = np.allclose(np.sort(topk_hat), true_topk)
    total_budget = np.sum(budgets).item()
    return is_correct, total_budget


def single_run_adasoftmax(
    sftm: SFTM,
    x: np.ndarray,
    k: int,
) -> Tuple[bool, int]:
    """
    Single run of the adaSoftmax algorithm.
    :returns: whether eps is in bounds, error, total budget
    """
    indices, z = sftm.softmax(x, k=k)
    indices_hat, z_hat, _ = sftm.adaptive_softmax(x, k=k)
    indices_hat = np.sort(indices_hat)
    assert(np.array_equal(indices, indices_hat))
    
    # test results
    error = np.abs(z_hat - z[indices]) / z[indices]
    in_bounds = error <= sftm.multiplicative_error  
    budget = np.sum(sftm.bandits.it)

    return in_bounds, error, budget


def epsilon_check(dataset, loader, **kwargs):
    """
    Runs adasoftmax once to check multiplicative error. 
    This test is for both mnl and llms.
    """
    # Load in A and xs
    if kwargs.get('is_mnl', False):  # defaults to False if not set
        A, xs = loader(dataset, testing=True)
    else:
        A, xs = loader(
            dataset,
            kwargs.get("model"),
            kwargs.get("stride"),
            testing=True
        )

    # Instantiate SFTM object
    n, d = A.shape
    sftm = SFTM(
        A,
        multiplicative_error=kwargs.get('eps'), 
        failure_probability=kwargs.get('delta'), 
        temperature=kwargs.get('temp'), 
        query_importance_sampling=kwargs.get('query_importance')
    )

    # Run test
    in_bounds, error, budget = single_run_adasoftmax(
        sftm=sftm,
        x=xs[0],
        k=kwargs.get('top_k'),
    )
    return in_bounds, budget, n * d


def delta_check(dataset, loader, **kwargs):
    """
    Runs adasoftmax num_experiment times to check delta error. 
    This test is for both mnl and llms.
    """
    # Load in A and xs
    if kwargs.get('is_mnl', False):  # defaults to False if not set
        A, xs = loader(dataset, testing=True)
    else:
        A, xs = loader(
            dataset,
            kwargs.get("model"),
            kwargs.get("stride"),
            testing=True
        )

    # Instantiate SFTM object
    n, d = A.shape
    sftm = SFTM(
        A, 
        multiplicative_error=kwargs.get('eps'), 
        failure_probability=kwargs.get('delta'), 
        temperature=kwargs.get('temp'), 
        query_importance_sampling=kwargs.get('query_importance')
    )
    
    # Run test
    total_wrong = 0
    total_budget = 0
    num_experiments = kwargs.get('num_experiments')  # Default value if not specified
    for i in range(min(xs.shape[0], num_experiments)):
        np.random.seed(i)

        # adasoftmax
        in_bounds, error, budget = single_run_adasoftmax(
            sftm=sftm,
            x=xs[i],
            k=kwargs.get('top_k'),
        )
        total_wrong += int(not in_bounds)
        total_budget += budget

    return total_wrong, total_budget, n * d * num_experiments