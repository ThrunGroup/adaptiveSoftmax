import numpy as np
from typing import Tuple
from math import log, ceil, sqrt
from scipy.special import logsumexp, softmax
from typing import Callable, Any

from adaptive_softmax.bandits_softmax import BanditsSoftmax
from adaptive_softmax.utils import fpc
from adaptive_softmax.constants import DEFAULT_CI_DECAY, TUNE_EXP_FUDGE_HIGH, TUNE_EXP_FUDGE_LOW

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
    self.seed = seed

    if self.verbose:
      print(f"Initializing SFTM for a matrix of shape ({self.n} x {self.d})...")
      print("Parameters:")
      print(f"\t- temperature: {self.temperature}")
      print(f"\t- multiplicative_error: {self.multiplicative_error}")
      print(f"\t- failure_probability: {self.failure_probability}")

    self.bandits = BanditsSoftmax(
      A,
      temperature=temperature,
      atom_importance_sampling=atom_importance_sampling,
      query_importance_sampling=query_importance_sampling,
      randomized_hadamard_transform=randomized_hadamard_transform,
      verbose=verbose,
      seed=self.seed,
    )

    if self.verbose:
      print("SFTM initialized.")
      print("")
  
  def tune_fudge_factors(self, X_train: np.ndarray, k: int=1) -> Tuple[float, float]:
    """
    Fits the fudge factors of SFTM based on the provided queries.

    @param X_train: The query vectors X_train of shape (b, d).
    @param k: The number of elements to return (default 1).
    @return: The fudge factors for the bandits and log norm estimation.
    """
    
    if self.verbose:
      print(f"Fitting SFTM fudge factors for {X_train.shape[0]} query vectors...")

    # get true best arms and log norm for each query
    MU = (X_train @ self.A.T) * self.temperature
    TOP_K = np.sort(np.argpartition(MU, -k, axis=1)[:, -k:], axis=1)
    LOG_NORM = logsumexp(MU, axis=1)

    delta = self.failure_probability
    eps = self.multiplicative_error

    # binary search for fudge factors
    def bin_search(f_check: Callable[[float, np.ndarray, np.ndarray, float], bool]) -> float:
      target_success_rate = 1 - delta  /2
      lo = TUNE_EXP_FUDGE_LOW
      hi = TUNE_EXP_FUDGE_HIGH
      while lo + 1e-2 < hi:
        mi = (lo + hi) / 2
        fudge_factor = 10 ** mi

        if self.verbose:
          print(f"\tTrying fudge factor: {fudge_factor}")

        count_correct = 0
        for i, x in enumerate(X_train):
          count_correct += f_check(fudge_factor, x, TOP_K[i], LOG_NORM[i])
        success_rate = count_correct / X_train.shape[0]

        if self.verbose:
          print(f"\tSuccess rate: {success_rate}")

        if success_rate < target_success_rate:
          lo = mi
        else:
          hi = mi

      return 10 ** hi
    
    def f_check_bandits(fudge_factor: float, x: np.ndarray, best_arms: np.ndarray, _: float) -> bool:
      self.bandits.set_query(x)
      best_arms_hat = self.best_arms(delta / 2, k, fudge_factor=fudge_factor)
      return np.all(best_arms_hat == best_arms)
    
    def f_check_log_norm(fudge_factor: float, x: np.ndarray, _: np.ndarray, log_norm: float) -> bool:
      self.bandits.set_query(x)
      log_norm_hat = self.log_norm_estimation(eps, delta / 2, fudge_factor=fudge_factor)
      return np.abs((log_norm_hat - log_norm) / log_norm) <= eps
    
    if self.verbose:
      print("Fitting bandits fudge factor...")

    fudge_bandits = bin_search(f_check_bandits)

    if self.verbose:
      print("Fitting log norm fudge factor...")

    fudge_log_norm = bin_search(f_check_log_norm)

    if self.verbose:
      print("Fitting complete.")
      print("")
    
    return fudge_bandits, fudge_log_norm

  def softmax(self, x: np.ndarray, k: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the true softmax, returning the top-k indices and the softmax.

    @param x: The query vector x of shape (d,).
    @param k: The number of elements to return (default 1).
    @return: The top-k indices and the softmax.
    """
    mu = (self.A @ x) * self.temperature
    top_k = np.sort(np.argpartition(mu, -k)[-k:])
    return top_k, softmax(mu)

  def adaptive_softmax(
      self,
      x: np.ndarray,
      k: int = 1,
      fudge_bandits: float = 1.0,
      fudge_log_norm: float = 1.0,
    ) -> Tuple[int, float]:
    """
    Computes the approximate softmax using the SFTM algorithm, returning the
    top-k indices, the approximate softmax for these indices, and the
    normalizing constant.

    This method is based on Algorithm 1 from the paper "Adaptive Sampling for
    Efficient Softmax Approximation."

    @param x: The query vector x of shape (d,).
    @param k: The number of elements to return (default 1).
    @param fudge_bandits: The multiplier for the variance estimate used in the
      bandits algorithm to account for loose bounds (default 1.0).
    @param fudge_log_norm: The multiplier for the variance estimate used in the
      log norm algorithm to account for loose bounds (default 1.0).
    @return: The top-k indices, the approximate softmax, and the normalizing
             constant Z.
    """

    if self.verbose:
      print(f"Computing adaptive softmax for query vector {x}...")

    self.bandits.set_query(x, seed=self.seed)

    eps = self.multiplicative_error
    delta = self.failure_probability
    sig2 = self.noise_bound if self.noise_bound is not None else self.bandits.variance

    if self.verbose:
      print(f"Noise bound: {sig2}")

    V0 = 1 / (17 * log(6 * self.n / delta))
    fudge_factor = max(fudge_bandits, fudge_log_norm)

    self.bandits.pull_to_var(
      np.arange(self.n), V0, fudge_factor_var=fudge_factor, batched=True)

    i_star_hat = self.best_arms(delta/2, k, fudge_factor=fudge_bandits)
    mu_star_hat = self.bandits.exact_values(i_star_hat)
    log_S_hat = self.log_norm_estimation(eps, delta/2, fudge_factor=fudge_log_norm)

    if self.verbose:
      print(f"Top-{k} arms: {i_star_hat}")
      print(f"Estimated logit values: {mu_star_hat}")
      print(f"Estimated log normalizing constant: {log_S_hat}")

    return i_star_hat, np.exp(mu_star_hat - log_S_hat), np.exp(log_S_hat)

  def best_arms(
      self,
      delta: float,
      k: int,
      ci_decay: float = DEFAULT_CI_DECAY,
      fudge_factor: float = 1.0,
    ) -> np.ndarray:
    """
    Finds the top-k arms with the highest estimated logit values.

    This method uses a round-based PAC bandits algorithm based on Algorithm 3
    from the paper, "Distributed Exploration in Multi-Armed Bandits" by Hillel
    et al. (2013).

    @param delta: The failure probability parameter.
    @param sig2: The noise bound parameter.
    @param k: The number of arms to return.
    @param ci_decay: The per-round decay factor for the confidence interval
      (default 0.5).
    @param fudge_factor: The multiplier for the variance estimate used in the
      bandits algorithm to account for loose bounds (default 1.0).
    @return: The top-k arms with the highest estimated logit values.
    """
    if self.verbose:
      print(f"Finding top-{k} arms with the highest estimated logit values...")

    n = self.n
    d = self.bandits.max_pulls
    v = 1 / (17 * log(6 * self.n / delta))

    # initialize parameters
    confidence_set = np.arange(n)

    while True:
      # pull arms and update confidence interval
      estimates, variances = self.bandits.pull_to_var(confidence_set, v, fudge_factor_var=fudge_factor)
      confidence_intervals = np.sqrt(2 * variances * log(6 * n * log(d) / delta))

      # update confidence set
      best_arm_hat = np.argmax(estimates)
      keep = estimates + confidence_intervals >= estimates[best_arm_hat] - confidence_intervals[best_arm_hat]

      if self.verbose:
        print(f"Confidence intervals: {confidence_intervals}")
        print(f"Estimates: {estimates}")
        print(f"Confidence set: {confidence_set[keep]}")

      # check stopping condition
      if np.sum(keep) <= k:
        break

      # update parameters
      confidence_set = confidence_set[keep]
      v *= ci_decay

    return confidence_set[np.argsort(estimates)[-k:]]

  def estimate_arm_logits(self, arms: np.ndarray, eps: float, delta: float, sig2: float) -> np.ndarray:
    """
    Estimates the logit values of the specified arms with PAC guarantees.

    The number of pulls needed to estimate the logit values of the specified
    arms with PAC guarantees if derived in Lemma 2 of the appendix of the SFTM
    paper.

    @param arms: The indices of the arms to estimate.
    @param eps: The multiplicative error parameter.
    @param delta: The failure probability parameter.
    @param sig2: The noise bound parameter.
    @return: The estimated logit values of the specified arms.
    """
    if self.verbose:
      print(f"Estimating logit values for arms {arms}...")

    d = self.bandits.max_pulls
    T = int(ceil(32 * sig2 * log(2 / delta) / (eps ** 2)))
    return self.bandits.pull(arms, its=np.array(fpc(T, d)))

  def log_norm_estimation(
      self,
      eps: float,
      delta: float,
      fudge_factor: float=1.0,
    ) -> float:
    """
    Estimates the log normalizing constant of the softmax function with PAC
    guarantees.

    This method is based on Algorithm 2 of the paper, "Adaptive Sampling for
    Efficient Softmax Approximation."

    @param eps: The multiplicative error parameter.
    @param delta: The failure probability parameter.
    @param fudge_factor: The multiplier for the variance estimate used in the
      log norm algorithm to account for loose bounds (default 1.0).
    @return: The estimated log normalizing constant of the softmax function.
    """

    n = self.n

    V0 = 1 / (17 * log(6 * n / delta))
    C = np.sqrt(2 * log(6 * n / delta) * V0)

    if self.verbose:
      print("Estimating log normalizing constant of the softmax function...")
      print(f"Initial sample mean to sample variance ratio: {V0}")
      print(f"Confidence interval constant: {C}")

    # initial estimates (should have already been done)
    mu_hat, _ = self.bandits.pull_to_var(np.arange(n), V0, fudge_factor_var=fudge_factor)

    if self.verbose:
      print(f"Initial estimates: {mu_hat}")

    log_alpha = (mu_hat - C)
    log_gamma = (mu_hat - C) / 2
    log_alpha_sum = logsumexp(log_alpha)
    log_gamma_sum = logsumexp(log_gamma)

    # adapt sample sizes based on initial estimates
    log_c = log(16 * sqrt(2) * log(6 * n / delta) / eps) + 2 * log_gamma_sum - log_alpha_sum
    log_d = log(16 * log(12 / delta) / (eps ** 2))

    V1 = np.full(n, V0)
    V1 = np.minimum(V1, np.exp(log_gamma_sum - (log_c + log_gamma)))
    V1 = np.minimum(V1, np.exp(log_alpha_sum - (log_d + log_alpha)))

    if self.verbose:
      print(f"Adaptive variance ratio thresholds: {V1}")

    # make updated estimates
    mu_hat, _ = self.bandits.pull_to_var(np.arange(n), V1, fudge_factor_var=fudge_factor)

    if self.verbose:
      print(f"Updated estimates: {mu_hat}")
      print(f"Estimated log normalizing constant: {logsumexp(mu_hat)}")

    return logsumexp(mu_hat)
    