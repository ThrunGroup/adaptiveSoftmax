import numpy as np
from typing import Tuple
from math import log, ceil, sqrt, exp
from scipy.special import logsumexp, softmax

from adaptive_softmax.bandits_softmax import BanditsSoftmax
from adaptive_softmax.utils import fpc

# TODO add constants from constants.py
# TODO add comments and verboseness from adasoftmax

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
    The temperature parameter for the softmax function, by default 1.0.
  multiplicative_error : float, optional
    The multiplicative error parameter for the PAC guarantee, by default 3e-1.
  failure_probability : float, optional
    The failure probability parameter for the PAC guarantee, by default 1e-1.
  noise_bound : float, optional
    The noise bound parameter for entries of the matrix-vector multiplication,
    by default None.
  fudge_pull : float, optional
    The multiplier for the number of pulls used in the bandits algorithm to 
    account for loose bounds, by default 1.0.
  fudge_sigma2 : float, optional
    The multiplier for the variance used in the bandits algorithm to account
    for loose bounds, by default 1.0.
  atom_importance_sampling : bool, optional
    The flag to enable atom-based importance sampling in the bandits algorithm,
    by default True.
  query_importance_sampling : bool, optional
    The flag to enable query-based importance sampling in the bandits algorithm,
    by default True.
  randomized_hadamard_transform : bool, optional
    The flag to enable randomized Hadamard transform of the atom matrix A
  verbose : bool, optional
    The flag to enable verbose output, by default False.
  seed : int, optional
    The seed for the random number generator used in the bandits algorithm, by
    default 42.
  """

  def __init__(self,
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
               seed=42):
    self.A = A
    self.n = A.shape[0]
    self.d = A.shape[1]
    self.temperature = temperature
    self.multiplicative_error = multiplicative_error
    self.failure_probability = failure_probability
    self.noise_bound = noise_bound

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

  def softmax(self, x: np.ndarray, k: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the true softmax, returning the top-k indices and the softmax.

    @param x: The query vector x of shape (d,).
    @param k: The number of elements to return, by default 1.
    @return: The top-k indices and the softmax.
    """
    mu = (self.A @ x) * self.temperature
    top_k = np.sort(np.argpartition(mu, -k)[-k:])
    return top_k, softmax(mu)

  def adaptive_softmax(self, x: np.ndarray, k: int=1) -> Tuple[int, float]:
    """
    Computes the approximate softmax using the SFTM algorithm, returning the
    top-k indices, the approximate softmax for these indices, and the
    normalizing constant.

    @param x: The query vector x of shape (d,).
    @param k: The number of elements to return, by default 1.
    @return: The top-k indices, the approximate softmax, and the normalizing
             constant.
    """
    self.bandits.set_query(x)

    bta = self.temperature
    eps = self.multiplicative_error
    dlt = self.failure_probability
    sig2 = self.noise_bound if self.noise_bound is not None else self.bandits.variance

    i_star_hat = self.best_arms(dlt/2, bta, sig2, k)
    mu_star_hat = self.bandits.exact_values(i_star_hat)
    log_S_hat = self.log_norm_estimation(bta, eps, dlt/2, sig2)

    return i_star_hat, np.exp(bta * mu_star_hat - log_S_hat), np.exp(log_S_hat)
  
  def best_arms(self, dlt: float, bta: float, sig2: float, k: int) -> np.ndarray:
    """
    Finds the top-k arms with the highest estimated logit values.

    This is a round-based PAC bandits algorithm: "Algorithm 3" from the paper 
    "Distributed Exploration in Multi-Armed Bandits" by Hillel et al. (2013).
    
    @param dlt: The failure probability parameter.
    @param bta: The temperature parameter.
    @param sig2: The noise bound parameter.
    @param k: The number of arms to return.
    @return: The top-k arms with the highest estimated logit values.
    """
    n = self.n
    d = self.bandits.max_pulls
    T0 = int(ceil(min(d, 17 * (bta ** 2) * sig2 * log(6 * n / dlt))))

    # initialize parameters
    confidence_set = np.arange(n)
    num_pulls = T0
    estimates = np.zeros(n)

    # TODO prevent infinite loop (for equl values) nicely

    while True:

      # pull arms and update confidence interval
      estimates = self.bandits.batch_pull(confidence_set, it=fpc(num_pulls, d))
      confidence_interval = sqrt(2 * sig2 * log(6 * n * log(d) / dlt) / num_pulls)

      # update confidence set
      keep = estimates >= np.max(estimates) - confidence_interval

      # check stopping condition
      if np.sum(keep) <= k:
        break

      # update parameters
      confidence_set = confidence_set[keep]
      num_pulls = num_pulls * 2

    return confidence_set[np.argsort(estimates)[-k:]]

  def estimate_arm_logits(self, arms: np.ndarray, bta: float, eps: float, dlt: float, sig2: float) -> np.ndarray:
    """
    Estimates the logit values of the specified arms with PAC guarantees.

    The number of pulls needed to estimate the logit values of the specified
    arms with PAC guarantees if derived in Lemma 2 of the appendix of the SFTM
    paper.

    @param arms: The indices of the arms to estimate.
    @param bta: The temperature parameter.
    @param eps: The multiplicative error parameter. 
    @param dlt: The failure probability parameter.
    @param sig2: The noise bound parameter.
    @return: The estimated logit values of the specified arms.
    """
    d = self.bandits.max_pulls
    T = int(ceil(min(d, 32 * (sig2) * (bta ** 2) * log(2 / dlt) / (eps ** 2))))
    return self.bandits.pull(arms, its=np.array(fpc(T, d)))
  
  def log_norm_estimation(self, bta: float, eps: float, dlt: float, sig2: float) -> float:
    """
    Estimates the log normalizing constant of the softmax function with PAC 
    guarantees.

    The process used to estimate the log normalizing constant of the softmax
    function with PAC guarantees is based on Algorithm 2 of the SFTM paper.

    @param bta: The temperature parameter.
    @param eps: The multiplicative error parameter.
    @param dlt: The failure probability parameter.
    @param sig2: The noise bound parameter.
    @return: The estimated log normalizing constant of the softmax function.
    """

    # initialize params and make initial estimates (lines 1-5)    
    n = self.n
    d = self.bandits.max_pulls

    T0 = int(ceil(min(d, 17 * (bta ** 2) * sig2 * log(6 * n / dlt))))
    C = np.sqrt(2 * sig2 * log(6 * n / dlt) / T0)
    
    mu_hat = self.bandits.pull(np.arange(n), its=np.full(shape=n, fill_value=fpc(T0, d)))

    log_alpha = bta * (mu_hat - C)
    log_gamma = bta * (mu_hat - C) / 2
    log_alpha_sum = logsumexp(log_alpha)
    log_gamma_sum = logsumexp(log_gamma)

    # make estimates (lines 6-7)
    log_b = log(17 * (bta ** 2) * sig2 * log(6 * n / dlt))
    log_c = log(16 * sqrt(2) * sig2 * log(6 * n / dlt) / eps) + 2 * log_gamma_sum - log_alpha_sum
    log_d = log(16 * sig2 * log(12 / dlt) / (eps ** 2))

    it = np.exp(log_b)
    it = np.maximum(it, np.exp(log_c + log_gamma - log_gamma_sum))
    it = np.maximum(it, np.exp(log_d + log_alpha - log_alpha_sum))
    it = np.minimum(it, d)
    it = np.ceil(it).astype(int)

    mu_hat = self.bandits.pull(np.arange(n), its=fpc(it, d))

    return logsumexp(bta * mu_hat)
    