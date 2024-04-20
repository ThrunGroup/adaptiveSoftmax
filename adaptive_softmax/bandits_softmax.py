import numpy as np
import torch
from hadamard_transform import hadamard_transform as ht

def generate_weighted_permutation(weights: np.ndarray, gen=np.random.default_rng(0)):
  assert np.all(weights >= 0)
  with np.errstate(divide='ignore'):
    logits = np.log(weights) - np.log(np.sum(weights))
    perturbed_logits = logits + gen.gumbel(size=logits.size)
    permutation = perturbed_logits.argsort()[::-1]
  return permutation, logits, perturbed_logits

class BanditsSoftmax:
  def __init__(
      self,
      A: np.ndarray,
      temperature: float = 1.0,
      atom_importance_sampling=True,
      query_importance_sampling=True,
      randomized_hadamard_transform=False,
      verbose=False,
      seed=42):
    
    assert len(A.shape) == 2

    self.n = A.shape[0]
    self.d = A.shape[1]
    self.temperature = temperature
    self.atom_importance_sampling = atom_importance_sampling
    self.query_importance_sampling = query_importance_sampling
    self.randomized_hadamard_transform = randomized_hadamard_transform
    self.verbose = verbose

    self._A = A
    self._x = None
    self._gen = np.random.default_rng(seed)

    if randomized_hadamard_transform:
      dp = 2 ** int(np.ceil(np.log2(self.d)))
      self._A = np.pad(A, ((0, 0), (0, dp - self.d)), 'constant', constant_values=0)
      self.d = dp
      self._rademacher = self._gen.choice([-1, 1], size=self.d)
      self._A = ht(torch.tensor(self._A * self._rademacher)).numpy()

    self._atom_weights = np.sum(np.abs(self._A), axis=0) if atom_importance_sampling else np.ones(self.d)
    self._permutation, self._logits, self._perturbed_logits = generate_weighted_permutation(self._atom_weights, gen=self._gen)
    
    # TODO deal with all-zero columns here

    q = (self._atom_weights / np.sum(self._atom_weights))[np.newaxis, :]
    self._est_atom_sig2 = np.max(np.sum((self._A / q / self.d) ** 2 * q, axis=1))
    self._est_query_sig2 = None
    self._sparse_columns = None

    self._Ap = None if self.query_importance_sampling else self._A[:, self._permutation].copy()
    self._xp = None

    self._it = np.zeros(self.n, dtype=int)
    self._estimates = np.zeros(self.n, dtype=np.float64)

    if self.verbose:
      print(f'BanditsSoftmax initialized with {self.n} arms and {self.d} dimensions')
      print(f'Atom importance sampling: {self.atom_importance_sampling}')
      print(f'Query importance sampling: {self.query_importance_sampling}')
      print(f'Randomized Hadamard transform: {self.randomized_hadamard_transform}')
      print(f'Permutation:\n{self._permutation}')
      if atom_importance_sampling:
        print(f'Atom weights:\n{self._atom_weights}')
      if randomized_hadamard_transform:
        print(f"Max variance before transform: {np.max(np.var(A, axis=1))}")
        print(f"Max variance after transform: {np.max(np.var(self._A, axis=1))}")
  
  @property
  def it(self):
    return self._it
  
  @property
  def max_pulls(self):
    assert self._x is not None

    return self.d - self._num_sparse_columns
  
  @property
  def variance(self):
    assert self._x is not None
    
    return self._est_atom_sig2 * self._est_query_sig2 * (self.max_pulls ** 2) # TODO change back to sparsity-based but with correct mean

  def set_query(self, x: np.ndarray):
    assert x.size <= self.d if self.randomized_hadamard_transform else x.size == self.d

    self._it = np.zeros(self.n, dtype=int)
    self._estimates = np.zeros(self.n, dtype=np.float64)

    self._x = np.pad(x, (0, self.d - x.size), 'constant', constant_values=0)

    if self.randomized_hadamard_transform:
      self._x = ht(torch.tensor(self._x * self._rademacher)).numpy()

    if self.query_importance_sampling:
      query_weights = np.abs(self._x)
      self._permutation, self._logits, self._perturbed_logits = generate_weighted_permutation(query_weights * self._atom_weights, gen=self._gen)
    
    self._xp = self._x[self._permutation].copy()

    self._num_sparse_columns = np.sum(np.isneginf(self._logits))
    n_nonzero = self.d - self._num_sparse_columns
    self._est_query_sig2 = np.mean(np.abs(self._xp[:n_nonzero])) ** 2 if self.query_importance_sampling else np.mean(self._xp[:n_nonzero] ** 2)

    if self.verbose:
      print(f'Query:\n{x}')
      if self.randomized_hadamard_transform:
        print(f'Query after Hadamard transform:\n{self._x}')
      if self.query_importance_sampling:
        print(f'Query weights:\n{query_weights}')
        print(f'Combined weights:\n{self._atom_weights * query_weights}')
        print(f'Permutation:\n{self._permutation}')
  
  def exact_values(self, arms: np.ndarray) -> np.ndarray:
    assert self._x is not None

    if np.any(self.it[arms] < self.max_pulls):
      A_arms = self._A[arms, self._permutation] if self._Ap is None else self._Ap[arms]
      self._estimates[arms] = (A_arms @ self._xp) * self.temperature
      self._it[arms] = self.max_pulls
    
    return self._estimates[arms]
  
  def pull_arm(self, arm: int, it: int) -> float:
    assert self._x is not None
    
    return self.batch_pull(np.atleast_1d(arm), it)[0]

  def pull(self, arms: np.ndarray, its: np.ndarray) -> np.ndarray:
    assert self._x is not None
    assert arms.size == its.size

    for i in np.nonzero(its > self._it[arms])[0]:
      self.batch_pull(np.atleast_1d(arms[i]), its[i])
    
    return self._estimates[arms]

  def batch_pull(self, arms: np.ndarray, it: int) -> np.ndarray:
    assert self._x is not None
    assert np.unique(self._it[arms]).size <= 1

    if arms.size == 0 or it <= self._it[0]:
      return self._estimates[arms]
    
    prev_it = self._it[arms][0]
    next_it = min(it, self.max_pulls)

    # importance sampling
    if self.atom_importance_sampling or self.query_importance_sampling:
      threshold = -np.inf if next_it == self.max_pulls else self._perturbed_logits[self._permutation[next_it]]
      weights = 1 - np.exp(-np.exp(self._logits[self._permutation[:next_it]] - threshold))
      weights = np.nan_to_num(weights, nan=1) # NOTE nan values have prob 0 and will never be selected

      A = (self._A[np.ix_(arms, self._permutation[:next_it])] if self._Ap is None else self._Ap[arms, :next_it]).reshape((len(arms), next_it))
      x = self._xp[:next_it] / weights
      self._estimates[arms] = (A @ x) * self.temperature

    # no importance sampling (equal weighting)
    else:
      self._estimates[arms] *= prev_it
      self._estimates[arms] += (self._Ap[arms, prev_it:next_it] @ self._xp[prev_it:next_it]) * (self.max_pulls * self.temperature)
      self._estimates[arms] /= next_it

    self._it[arms] = next_it
    return self._estimates[arms]
