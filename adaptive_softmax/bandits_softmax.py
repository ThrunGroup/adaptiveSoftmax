import numpy as np
import torch
from hadamard_transform import randomized_hadamard_transform

# TODO tensor-ify
# TODO estimate sigma^2
# TODO add documentation

def generate_weighted_permutation(weights: np.ndarray, gen=np.random.default_rng(0)):
  assert np.all(weights >= 0)
  with np.errstate(divide='ignore'):
    logits = np.log(weights) - np.log(np.sum(weights))
    perturbed_logits = logits + gen.gumbel(size=len(logits))
    permutation = perturbed_logits.argsort()[::-1]
  return permutation, logits, perturbed_logits

class BanditsSoftmax:
  def __init__(
      self,
      A: np.ndarray,
      atom_importance_sampling=True,
      query_importance_sampling=True,
      randomized_hadamard_transform=True,
      verbose=False,
      seed=42):
    
    assert len(A.shape) == 2

    self.n = A.shape[0]
    self.d = A.shape[1]
    self.atom_importance_sampling = atom_importance_sampling
    self.query_importance_sampling = query_importance_sampling
    self.randomized_hadamard_transform = randomized_hadamard_transform
    self.verbose = verbose

    self._A = A
    self._x = None
    self._gen = np.random.default_rng(seed)
    self._torch_gen = torch.Generator()
    self._seed = seed

    if randomized_hadamard_transform:
      dp = 2 ** int(np.ceil(np.log2(self.d)))
      self._A = np.pad(A, ((0, 0), (0, dp - self.d)), 'constant', constant_values=0, seed=seed)
      self._A = randomized_hadamard_transform(torch.from_numpy(self._A), prng=self._torch_gen.manual_seed(self._seed)).numpy()
      self.d = dp

    self._atom_weights = np.sum(np.abs(self._A), axis=0) if atom_importance_sampling else np.ones(self.d)
    self._permutation, self._logits, self._perturbed_logits = generate_weighted_permutation(self._atom_weights, gen=self._gen)

    self._Ap = None if self.query_importance_sampling else self._A[self._permutation].copy()
    self._xp = None

    self._it = np.zeros(self.n)
    self._estimates = np.zeros(self.n)

    if self.verbose:
      print(f'BanditsSoftmax initialized with {self.n} arms and {self.d} dimensions')
      print(f'Atom importance sampling: {self.atom_importance_sampling}')
      print(f'Query importance sampling: {self.query_importance_sampling}')
      print(f'Randomized Hadamard transform: {self.randomized_hadamard_transform}')
      print(f"Empirical sigma^2: {self._sigma2_atom}")
      print(f'Permutation:\n{self._permutation}')
      if atom_importance_sampling:
        print(f'Atom weights:\n{self._atom_weights}')
      if randomized_hadamard_transform:
        print(f'Empirical sigma^2 prior to Hadamard transform: {np.median(np.var(A), axis=0, ddof=1)}')
  
  @property
  def it(self):
    return self._it

  def set_query(self, x: np.ndarray):
    assert len(x) <= self.d if self.randomized_hadamard_transform else len(x) == self.d

    self._x = np.pad(x, (0, self.d - len(x)), 'constant', constant_values=0)

    if self.randomized_hadamard_transform:
      self._x = randomized_hadamard_transform(torch.from_numpy(self._x.T), prng=self._torch_gen.manual_seed(self._seed)).numpy().T

    if self.query_importance_sampling:
      query_weights = np.abs(self._x)
      self._permutation, self._logits, self._perturbed_logits = generate_weighted_permutation(self._atom_weights * query_weights, gen=self._gen)
    
    self._xp = self._x[self._permutation].copy()

    if self.verbose:
      print(f'Query:\n{self._x}')
      if self.query_importance_sampling:
        print(f'Query weights:\n{query_weights}')
        print(f'Combined weights:\n{self._atom_weights * query_weights}')
        print(f'Permutation:\n{self._permutation}')
  
  def pull(self, arm: int, it: int) -> float:
    assert self._x is not None
    
    return self.batch_pull(np.array(arm), it)[0]

  def pull(self, arms: np.ndarray, its: np.ndarray) -> np.ndarray:
    assert self._x is not None
    assert len(arms) == len(its)

    for i in np.nonzero(its > self._it[arms])[0]:
      self.batch_pull(np.array(arms[i]), its[i])
    
    return self._estimates[arms]

  def batch_pull(self, arms: np.ndarray, it: int) -> np.ndarray:
    assert self._x is not None
    assert len(np.unique(self._it[arms])) <= 1

    if len(arms) == 0 or it <= self._it[0]:
      return self._estimates[arms]
    
    prev_it = self._it[arms][0]
    next_it = min(it, self.d)

    # importance sampling
    if self.atom_importance_sampling or self.query_importance_sampling:
      threshold = -np.inf if next_it == self.d else self._perturbed_logits[self._permutation[next_it]]
      weights = 1 - np.exp(-np.exp(self._logits[self._permutation[:next_it]] - threshold))
      A = self._A[arms, self._permutation[:next_it]] if self._Ap is None else self._Ap[arms, :next_it]
      x = self._xp[:next_it] / weights
      self._estimates[arms] = (A @ x) / next_it

    # no importance sampling (equal weighting)
    else:
      self._estimates[arms] *= prev_it
      self._estimates[arms] += (self._Ap[arms, prev_it:next_it] @ self._xp[prev_it:next_it]) * self.d
      self._estimates[arms] /= next_it

    self._it[arms] = next_it
    return self._estimates[arms]
