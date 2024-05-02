import numpy as np
from typing import Tuple
from math import log, ceil, sqrt
from scipy.special import logsumexp, softmax

from adaptive_softmax.bandits_softmax import BanditsSoftmax
from adaptive_softmax.utils import fpc

class SFTM:
  """
  Softmax Fast Top-k via Monte Carlo (SFTM) approximates the softmax function
  following a matrix-vector multiplication looking at only a subset of the
  columns of the matrix and with provable PAC guarantees. See [paper link] for 
  more details.

  Parameters
  ----------
  A : np.ndarray
    The atom matrix A of shape (n, d) for the matrix-vector multiplication.
  temperature : float, optional
    The temperature parameter for the softmax function (default 1.0).
  multiplicative_error : float, optional
    The multiplicative error parameter for the PAC guarantee, epsilon (default 3e-1).
  failure_probability : float, optional
    The failure probability parameter for the PAC guarantee, delta (default 1e-1).
  noise_bound : float, optional
    The noise bound parameter for entries of the matrix-vector multiplication (default None).
  fudge_pull : float, optional
    The multiplier for the number of pulls used in the bandits algorithm to 
    account for loose bounds (default 1.0).
  fudge_sigma2 : float, optional
    The multiplier for the variance used in the bandits algorithm to account
    for loose bounds (default 1.0).
  atom_importance_sampling : bool, optional
    The flag to enable atom-based importance sampling in the bandits algorithm (default True).
  query_importance_sampling : bool, optional
    The flag to enable query-based importance sampling in the bandits algorithm (default True).
  randomized_hadamard_transform : bool, optional
    The flag to enable randomized Hadamard transform of the atom matrix A (default False)
  verbose : bool, optional
    The flag to enable verbose output (default False).
  seed : int, optional
    The seed for the random number generator used in the bandits algorithm (default 42).
  """

  def __init__(
     self,
     A: np.ndarray,
     temperature: float = 1.0,
     multiplicative_error: float = 3e-1,
     failure_probability: float = 1e-1,
     noise_bound: float = None,
     fudge_pull: float = 1.0,
     fudge_sigma2: float = 1.0,
     atom_importance_sampling: bool = True,
     query_importance_sampling: bool = True,
     randomized_hadamard_transform: bool = False,
     verbose: bool = False,
     seed=42
    ):
    self.A = A
    self.n = A.shape[0]
    self.d = A.shape[1]
    self.temperature = temperature
    self.multiplicative_error = multiplicative_error
    self.failure_probability = failure_probability
    self.noise_bound = noise_bound
    self.verbose = verbose

    if self.verbose:
      print(f"Initializing SFTM for a matrix of shape ({self.n} x {self.d})...")
      print("Parameters:")
      print(f"\t- temperature: {self.temperature}")
      print(f"\t- multiplicative_error: {self.multiplicative_error}")
      print(f"\t- failure_probability: {self.failure_probability}")

    self.bandits = BanditsSoftmax(
      A,
      temperature=temperature,
      fudge_pull=fudge_pull,
      fudge_sigma2=fudge_sigma2,
      atom_importance_sampling=atom_importance_sampling,
      query_importance_sampling=query_importance_sampling,
      randomized_hadamard_transform=randomized_hadamard_transform,
      verbose=verbose,
      seed=seed,
    )

    if self.verbose:
      print("SFTM initialized.")
      print("")

  def softmax(self, x: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the true softmax, returning the top-k indices and the softmax.

    @param x: The query vector x of shape (d,).
    @param k: The number of elements to return (default 1).
    @return: The top-k indices and the softmax.
    """
    mu = (self.A @ x) * self.temperature
    top_k = np.sort(np.argpartition(mu, -k)[-k:])
    return top_k, softmax(mu)

  def adaptive_softmax(self, x: np.ndarray, k: int = 1) -> Tuple[int, float]:
    """
    Computes the approximate softmax using the SFTM algorithm, returning the
    top-k indices, the approximate softmax for these indices, and the
    normalizing constant.

    This method is based on Algorithm 1 from the paper "Adaptive Sampling for
    Efficient Softmax Approximation."

    @param x: The query vector x of shape (d,).
    @param k: The number of elements to return (default 1).
    @return: The top-k indices, the approximate softmax, and the normalizing
             constant Z.
    """

    if self.verbose:
      print(f"Computing adaptive softmax for query vector {x}...")

    self.bandits.set_query(x)

    beta = self.temperature
    eps = self.multiplicative_error
    delta = self.failure_probability
    sig2 = self.noise_bound if self.noise_bound is not None else self.bandits.variance

    if self.verbose:
      print(f"Noise bound: {sig2}")

    # TODO(@colin): Did we decide whether this should be delta/2 or delta?
    i_star_hat = self.best_arms(delta/2, beta, sig2, k)
    # TODO(@colin): if i_star_hat is wrong, won't mu_star_hat also be the wrong value?
    mu_star_hat = self.bandits.exact_values(i_star_hat)
    # TODO(@colin): Did we decide whether this should be delta/2 or delta?
    log_S_hat = self.log_norm_estimation(beta, eps, delta/2, sig2)

    if self.verbose:
      print(f"Top-{k} arms: {i_star_hat}")
      print(f"Estimated logit values: {mu_star_hat}")
      print(f"Estimated log normalizing constant: {log_S_hat}")

    return i_star_hat, np.exp(beta * mu_star_hat - log_S_hat), np.exp(log_S_hat)

  def best_arms(self, delta: float, beta: float, sig2: float, k: int) -> np.ndarray:
    """
    Finds the top-k arms with the highest estimated logit values.

    This method uses a round-based PAC bandits algorithm based on Algorithm 3
    from the paper, "Distributed Exploration in Multi-Armed Bandits" by Hillel
    et al. (2013).

    @param delta: The failure probability parameter.
    @param beta: The temperature parameter.
    @param sig2: The noise bound parameter.
    @param k: The number of arms to return.
    @return: The top-k arms with the highest estimated logit values.
    """
    if self.verbose:
      print(f"Finding top-{k} arms with the highest estimated logit values...")

    n = self.n
    d = self.bandits.max_pulls
    T0 = int(ceil(17 * (beta ** 2) * sig2 * log(6 * n / delta)))

    if self.verbose:
      print(f"Initial number of pulls: {T0}")

    # initialize parameters
    confidence_set = np.arange(n)
    num_pulls = T0
    keep_pulling = True
    while keep_pulling is True:
      # pull arms and update confidence interval
      estimates = self.bandits.batch_pull(confidence_set, it=fpc(num_pulls, d))
      confidence_interval = sqrt(2 * sig2 * log(6 * n * log(d) / delta) / num_pulls)

      # update confidence set
      keep = estimates >= np.max(estimates) - confidence_interval

      # TODO(@colin): I don't think this is exactly correct. It may be the case that an arm is
      #  removed at some point, but then np.max(estimates) moves down and the arm gets added back later.
      #  The current implementation would say that arm has been pulled num_pulls times, but it's been pulled
      #  fewer times. For this reason, I think it's actually better to make num_pulls an *array* of how many
      #  times each arm has been pulled, and then update the confidence interval for each arm separately according
      #  to its number of pulls. This is how we did it in several other projects, see BanditPAM, FastForest, or BanditMIPS
      #  for examples.

      if self.verbose:
        print(f"Number of pulls: {num_pulls}")
        print(f"FPC-adjusted number of pulls: {fpc(num_pulls, d)}")
        print(f"Estimates: {estimates}")
        print(f"Confidence interval: {confidence_interval}")
        print(f"Confidence set: {confidence_set[keep]}")

      # check stopping condition
      if np.sum(keep) <= k or fpc(num_pulls, d) >= d:
        keep_pulling = False
        break

      # update parameters
      confidence_set = confidence_set[keep]
      num_pulls = num_pulls * 2

    return confidence_set[np.argsort(estimates)[-k:]]

  def estimate_arm_logits(self, arms: np.ndarray, beta: float, eps: float, delta: float, sig2: float) -> np.ndarray:
    """
    Estimates the logit values of the specified arms with PAC guarantees.

    The number of pulls needed to estimate the logit values of the specified
    arms with PAC guarantees if derived in Lemma 2 of the appendix of the SFTM
    paper.

    @param arms: The indices of the arms to estimate.
    @param beta: The temperature parameter.
    @param eps: The multiplicative error parameter.
    @param delta: The failure probability parameter.
    @param sig2: The noise bound parameter.
    @return: The estimated logit values of the specified arms.
    """
    if self.verbose:
      print(f"Estimating logit values for arms {arms}...")

    d = self.bandits.max_pulls
    T = int(ceil(32 * sig2 * (beta ** 2) * log(2 / delta) / (eps ** 2)))
    return self.bandits.pull(arms, its=np.array(fpc(T, d)))

  def log_norm_estimation(self, beta: float, eps: float, delta: float, sig2: float) -> float:
    """
    Estimates the log normalizing constant of the softmax function with PAC
    guarantees.

    This method is based on Algorithm 2 of the paper, "Adaptive Sampling for
    Efficient Softmax Approximation."

    @param beta: The temperature parameter.
    @param eps: The multiplicative error parameter.
    @param delta: The failure probability parameter.
    @param sig2: The noise bound parameter.
    @return: The estimated log normalizing constant of the softmax function.
    """

    n = self.n
    d = self.bandits.max_pulls

    T0 = int(ceil(17 * (beta ** 2) * sig2 * log(6 * n / delta)))
    C = np.sqrt(2 * sig2 * log(6 * n / delta) / T0)

    if self.verbose:
      print("Estimating log normalizing constant of the softmax function...")
      print(f"Initial number of pulls: {T0}")
      print(f"Confidence interval constant: {C}")

    # initial estimates
    mu_hat = self.bandits.pull(np.arange(n), its=np.full(shape=n, fill_value=fpc(T0, d)))

    if self.verbose:
      print(f"Initial estimates: {mu_hat}")

    log_alpha = beta * (mu_hat - C)
    log_gamma = beta * (mu_hat - C) / 2
    log_alpha_sum = logsumexp(log_alpha)
    log_gamma_sum = logsumexp(log_gamma)

    # adapt sample sizes based on initial estimates
    log_b = log(17 * (beta ** 2) * sig2 * log(6 * n / delta))
    log_c = log(16 * sqrt(2) * sig2 * log(6 * n / delta) / eps) + 2 * log_gamma_sum - log_alpha_sum
    log_d = log(16 * sig2 * log(12 / delta) / (eps ** 2))

    it = np.exp(log_b)
    it = np.maximum(it, np.exp(log_c + log_gamma - log_gamma_sum))
    it = np.maximum(it, np.exp(log_d + log_alpha - log_alpha_sum))
    it = np.ceil(it).astype(int)

    if self.verbose:
      print(f"Adaptive sample sizes: {it}")

    # make updated estimates
    mu_hat = self.bandits.pull(np.arange(n), its=fpc(it, d))

    if self.verbose:
      print(f"Updated estimates: {mu_hat}")
      print(f"Estimated log normalizing constant: {logsumexp(beta * mu_hat)}")

    return logsumexp(beta * mu_hat)
    