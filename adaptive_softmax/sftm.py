import numpy as np
from typing import Tuple
from math import log, ceil, sqrt, exp
from scipy.special import logsumexp, softmax

from adaptive_softmax.bandits_softmax import BanditsSoftmax

# TODO add documentation
# TODO add comments to assertions
# TODO add constants from constants.py
# TODO add comments and verboseness from adasoftmax
# TODO estimate sigma^2
# TODO use sftm in experiments

class SFTM:
  def __init__(self,
               A: np.ndarray,
               temperature: float = 1.0,
               multiplicative_error: float = 3e-1,
               failure_probability: float = 1e-1,
               noise_bound: float = 1,
               atom_importance_sampling: bool = True,
               query_importance_sampling: bool = True,
               randomized_hadamard_transform: bool = True,
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
      atom_importance_sampling=atom_importance_sampling,
      query_importance_sampling=query_importance_sampling,
      randomized_hadamard_transform=randomized_hadamard_transform,
      verbose=verbose,
      seed=seed,
    )

    # NOTE the matrix A may be transformed to reduce variance
    self.max_pulls = self.bandits.d

  def softmax(self, x: np.ndarray, k: int=1) -> np.ndarray:
    mu = self.A @ x
    top_k = np.sort(np.argpartition(mu, -k)[-k:])
    return top_k, softmax(mu)

  def adaptive_softmax(self, x: np.ndarray, k: int=1) -> Tuple[int, float]:
    bta = self.temperature
    eps = self.multiplicative_error
    dlt = self.failure_probability
    sig2 = self.noise_bound

    self.bandits.set_query(x)

    # i_star_hat = self.best_arm(x, dlt/3, bta, sig2)
    # mu_star_hat = self.estimate_arm_logit(x, i_star_hat, bta, eps/4, dlt/3, sig2)
    # log_S_hat = self.log_norm_estimation(x, bta, eps/4, dlt/3, sig2)

    i_star_hat = self.best_arms(dlt/2, bta, sig2, k)
    mu_star_hat = self.bandits.exact_values(i_star_hat)
    log_S_hat = self.log_norm_estimation(bta, eps, dlt/2, sig2)

    return i_star_hat, np.exp(bta * mu_star_hat - log_S_hat), np.exp(log_S_hat)
  
  # Algorithm 3 (https://proceedings.neurips.cc/paper_files/paper/2013/file/598b3e71ec378bd83e0a727608b5db01-Paper.pdf)
  def best_arms(self, dlt: float, bta: float, sig2: float, k: int) -> int:
    n = self.n
    d = self.max_pulls
    T0 = int(ceil(min(d, 17 * (bta ** 2) * sig2 * log(6 * n / dlt))))

    # initialize parameters
    confidence_set = np.arange(n)
    num_pulls = T0
    estimates = np.zeros(n)

    while True:
      # pull arms and update confidence interval
      estimates = self.bandits.batch_pull(confidence_set, it=num_pulls)
      confidence_interval = sqrt(2 * sig2 * log(6 * n * log(d) / dlt) / num_pulls)
      
      # finite population correction
      confidence_interval *= np.sqrt((d - num_pulls) / (d - 1))

      # update confidence set
      keep = estimates >= np.max(estimates) * (1 - confidence_interval)
      #keep = estimates >= np.max(estimates) - confidence_interval

      # check stopping condition
      if np.sum(keep) <= k:
        break

      # update parameters
      confidence_set = confidence_set[keep]
      num_pulls = min(d, num_pulls * 2)

    return confidence_set[np.argsort(estimates)[-k:]]

  # Appendix Lemma 2 (Exponential best arm estimation)
  def estimate_arm_logits(self, arms: np.ndarray, bta: float, eps: float, dlt: float, sig2: float) -> float:
    T = int(ceil(min(self.max_pulls, 32 * (sig2) * (bta ** 2) * log(2 / dlt) / (eps ** 2))))
    return self.bandits.pull(arms, its=np.array(T))
  
  # Algorithm 2 (Normalization estimation)
  def log_norm_estimation(self, bta: float, eps: float, dlt: float, sig2: float) -> float:
    # initialize params and make initial estimates (lines 1-5)    
    n = self.n
    d = self.max_pulls

    T0 = int(ceil(min(d, 17 * (bta ** 2) * sig2 * log(6 * n / dlt))))
    C = np.sqrt(2 * sig2 * log(6 * n / dlt) / T0)
    
    mu_hat = self.bandits.pull(np.arange(n), its=np.full(shape=n, fill_value=T0))

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

    mu_hat = self.bandits.pull(np.arange(n), its=it)

    return logsumexp(bta * mu_hat)
    