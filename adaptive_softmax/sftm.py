import numpy as np
from typing import Tuple
from math import log, ceil, sqrt
from scipy.special import logsumexp, softmax

# TODO convert iter to total its across alg
# TODO integrate arm puller into SFTM
# TODO add documentation
# TODO pull in Ryan's code
# TODO change all references to adasoftmax
# TODO add tests

class SFTM:
  def __init__(self,
               A: np.ndarray,
               temperature: float = 1.0,
               multiplicative_error: float = 3e-1,
               failure_probability: float = 1e-2,
               noise_bound: float = 1e1,
               seed=42):
    self.A = A
    self.n = A.shape[0]
    self.d = A.shape[1]
    self.temperature = temperature
    self.multiplicative_error = multiplicative_error
    self.failure_probability = failure_probability
    self.noise_bound = noise_bound
    self.gen = np.random.Generator(np.random.PCG64(seed))
  
  def softmax(self, x: np.ndarray):
    return softmax(self.A @ x * self.temperature)
    # logits = (self.A @ x) * self.temperature
    # logits -= np.max(logits)
    # w = np.exp(logits)
    # return w / np.sum(w)

  def adaptive_softmax(self, x: np.ndarray) -> Tuple[int, float]:
    bta = self.temperature
    eps = self.multiplicative_error
    dlt = self.failure_probability
    sig2 = self.noise_bound

    # i_star_hat = self.best_arm(x, dlt/3, bta, sig2)
    # mu_star_hat = self.estimate_arm_logit(x, i_star_hat, bta, eps/4, dlt/3, sig2)
    # log_S_hat = self.log_norm_estimation(x, bta, eps/4, dlt/3, sig2)

    i_star_hat = self.best_arm(x, dlt/2, bta, sig2)
    mu_star_hat = self.A[i_star_hat] @ x # NOTE true value for best arm only
    log_S_hat = self.log_norm_estimation(x, bta, eps, dlt/2, sig2)

    return i_star_hat, np.exp(bta * mu_star_hat - log_S_hat)
  
  # Algorithm 3 (https://proceedings.neurips.cc/paper_files/paper/2013/file/598b3e71ec378bd83e0a727608b5db01-Paper.pdf)
  def best_arm(self, x: np.ndarray, dlt: float, bta: float, sig2: float) -> int:
    # initialize params (line 1)
    n = self.n
    r = 0
    S_r = np.arange(n)
    t_rp = 0
    t_r = min(self.d, 17 * (bta ** 2) * sig2 * log(6 * n / dlt))
    eps_r = 1
    p = np.zeros(n)
    it = 0
    while True:
      # update parameters (lines 3-4)
      r += 1
      eps_r /= 2
      # t_r = (2 / (eps_r ** 2)) * log(4 * n * (r ** 2) / dlt)
      it_r = min(self.d, t_r - t_rp)
      t_rp = t_r
      t_r *= 2

      # pull arms and update estimates (lines 5-9)
      mu_est_r = self.pull_arms(S_r, x, it=it_r)
      p = np.average(np.vstack((p, mu_est_r)), axis=0, weights=np.array([it, it_r]))
      it += it_r

      # update confidence set (lines 10-11)
      p_max = np.max(p)
      keep = p >= p_max * (1-eps_r)
      S_r = S_r[keep]
      p = p[keep]

      # check stopping condition (line 12)
      if len(S_r) == 1:
        break

    return S_r[np.argmax(p)]

  # Appendix Lemma 2 (Exponential best arm estimation)
  def estimate_arm_logit(self, x: np.ndarray, arm: int, bta: float, eps: float, dlt: float, sig2: float) -> float:
    T = min(self.d, 32 * (sig2) * (bta ** 2) * log(2 / dlt) / (eps ** 2))
    return self.pull_arms(np.array([arm]), x, it=T)
  
  # Algorithm 2 (Normalization estimation)
  def log_norm_estimation(self, x: np.ndarray, bta: float, eps: float, dlt: float, sig2: float) -> float:
    # initialize params and make initial estimates (lines 1-5)    
    n = self.n
    T0 = min(self.d, 17 * (bta ** 2) * sig2 * log(6 * n / dlt))
    mu_hat = self.pull_arms(np.arange(n), x, it=T0)
    C = np.sqrt(2 * sig2 * log(6 * n / dlt) / T0)
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
    it -= T0  # NOTE this is not in the paper
    it = np.minimum(it, self.d)

    for arm in np.where(it >= 1)[0]:
      mu_hat[arm] *= T0
      mu_hat[arm] += self.pull_arms(np.array([arm]), x, it=it[arm]) * it[arm]
      mu_hat[arm] /= T0 + it[arm]

    return logsumexp(bta * mu_hat)

  # NOTE uses same column as pull for each arm
  def pull_arms(self, arms: np.ndarray, x: np.ndarray, it: float=1) -> np.ndarray:
    it = int(ceil(it))
    # j = self.gen.choice(self.d, size=it, replace=False)
    # i = arms
    # return self.A[np.ix_(i, j)] * x[j] * self.d
    return (self.d / it) * (self.A[arms, :it] @ x[:it]) # TODO remove (for profiling)
    