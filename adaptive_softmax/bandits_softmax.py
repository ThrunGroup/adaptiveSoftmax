import numpy as np
import torch
from hadamard_transform import hadamard_transform as ht
from math import ceil
from typing import Union, Tuple
from adaptive_softmax.constants import DEFAULT_VAR_PULL_INIT, DEFAULT_VAR_PULL_INCR

def generate_weighted_permutation(weights: np.ndarray, gen=np.random.default_rng(0)):
  """
  Generate a weighed permutation using the Gumbel trick. Any size-k prefix of
  this permutation represents a weighted reservoir sample of size k.

  @param weights: The non-negative weights to use for the permutation
  @param gen: The random number generator to use
  @return: The permutation, the logits, and the perturbed logits
  """
  assert np.all(weights >= 0), 'Weights must be non-negative'

  with np.errstate(divide='ignore'):
    logits = np.log(weights) - np.log(np.sum(weights))
    perturbed_logits = logits + gen.gumbel(size=logits.size)
    permutation = perturbed_logits.argsort()[::-1]

  return permutation, logits, perturbed_logits

class BanditsSoftmax:
  """
  A class to handle the bandit problem used to perform adaptive softmax.

  This class performs pre-computation based on the provided atoms to reduce the 
  variance of the resulting arm pulls. Once a query is provided, the class can
  handle arm pulls and the resulting updates to the estimated mean of each 
  bandit arm for the adaptive softmax computation. The class is meant to be used
  in conjunction with the SFTM class.

  Parameters
  ----------
  A : np.ndarray
    The atom matrix A of shape (n, d) for the matrix-vector multiplication
  temperature : float, optional
    The temperature of the softmax (default 1.0)
  atom_importance_sampling : bool, optional
    The flag to enable atom-based importance sampling in the bandits algorithm
      (default True)
  query_importance_sampling : bool, optional
    The flag to enable query-based importance sampling in the bandits algorithm
      (default True)
  randomized_hadamard_transform : bool, optional
    The flag to enable randomized Hadamard transform of the atoms (default False)
  verbose : bool, optional
    The flag to enable verbose output (default False)
  seed : int, optional
    The seed for the random number generator (default 42)
  """

  def __init__(
      self,
      A: np.ndarray,
      temperature: float = 1.0,
      atom_importance_sampling=True,
      query_importance_sampling=True,
      randomized_hadamard_transform=False,
      verbose=False,
      seed=42,
  ):
    assert len(A.shape) == 2, 'A must be a 2D array'

    self.n = A.shape[0]
    self.d = A.shape[1]
    self.temperature = temperature
    self.atom_importance_sampling = atom_importance_sampling
    self.query_importance_sampling = query_importance_sampling
    self.randomized_hadamard_transform = randomized_hadamard_transform
    self.verbose = verbose

    self._A = A
    self._x = None
    
    gen = np.random.default_rng(seed)

    if randomized_hadamard_transform:
      dp = 2 ** int(np.ceil(np.log2(self.d)))
      self._A = np.pad(A, ((0, 0), (0, dp - self.d)), 'constant', constant_values=0)
      self.d = dp
      self._rademacher = gen.choice([-1, 1], size=self.d)
      self._A = ht(torch.tensor(self._A * self._rademacher)).numpy()

    self._atom_weights = np.sum(np.abs(self._A), axis=0) if atom_importance_sampling else np.ones(self.d)
    self._permutation, self._logits, self._perturbed_logits = generate_weighted_permutation(self._atom_weights, gen=gen)
    
    q = (self._atom_weights / (np.sum(self._atom_weights)) )[np.newaxis, :]
    q[q == 0 | np.isnan(q)] = 1  # NOTE 0-weight columns will never be selected
    self._est_atom_sig2 = np.max(np.sum((self._A / q / self.d) ** 2 * q, axis=1))
    self._est_query_sig2 = None
    self._sparse_columns = None

    self._Ap = None if self.query_importance_sampling else self._A[:, self._permutation].copy()
    self._xp = None

    self._it = np.zeros(self.n, dtype=int)
    self._estimates = np.zeros(self.n, dtype=np.float64)
    self._var = np.full(self.n, np.inf, dtype=np.float64)

    if self.verbose:
      print(f'BanditsSoftmax initialized with {self.n} arms and {self.d} dimensions')
      print(f'Atom importance sampling: {self.atom_importance_sampling}')
      print(f'Query importance sampling: {self.query_importance_sampling}')
      print(f'Randomized Hadamard transform: {self.randomized_hadamard_transform}')
      print(f'Permutation:\n{self._permutation}')

      if atom_importance_sampling:
        print(f'Atom weights:\n{self._atom_weights}')

      if randomized_hadamard_transform:
        print(f'Columns 0-padded: {A.shape[1]} --> {self.d}')
  
  @property
  def it(self):
    """
    The number of pulls for each arm.
    """
    return self._it
  
  @property
  def max_pulls(self):
    """
    The maximum number of times any arm can be pulled.

    This may be different from the different dimension of the provided atoms if
    padding was performed for the randomized Hadamard transform or if the query
    vector is sparse and query-based importance sampling is enabled.
    """
    assert self._x is not None, 'Query vector not set'

    return self.d - self._num_sparse_columns
  
  @property
  def variance(self):
    """
    An upper bound of the variance of the bandit pulls.
    """
    assert self._x is not None, 'Query vector not set'
    
    return self._est_atom_sig2 * self._est_query_sig2 * (self.max_pulls ** 2) * (self.temperature ** 2)

  def set_query(self, x: np.ndarray, seed=42):
    """
    Set the query vector for the bandit problem.

    This method prepares the bandit problem for the provided query vector. The
    query vector is padded with zeros if necessary and transformed using a
    randomized Hadamard transform if enabled. If query-based importance sampling
    is enabled, the query weights are computed based on the magnitude of entries
    and the order in which column arms are pulled is re-sampled.

    @param x: The query vector
    """
    assert x.size <= self.d if self.randomized_hadamard_transform else x.size == self.d, 'Query vector must of of size d or less if padding was performed due to a randomized Hadamard transform'

    gen = np.random.default_rng(seed)

    self._it = np.zeros(self.n, dtype=int)
    self._estimates = np.zeros(self.n, dtype=np.float64)
    self._var = np.full(self.n, np.inf, dtype=np.float64)

    self._x = np.pad(x, (0, self.d - x.size), 'constant', constant_values=0)

    if self.randomized_hadamard_transform:
      self._x = ht(torch.tensor(self._x * self._rademacher)).numpy()

    if self.query_importance_sampling:
      query_weights = np.abs(self._x)
      self._permutation, self._logits, self._perturbed_logits = generate_weighted_permutation(query_weights * self._atom_weights, gen=gen)
    
    self._xp = self._x[self._permutation].copy()

    self._num_sparse_columns = np.sum(np.isneginf(self._logits))
    n_nonzero = self.d - self._num_sparse_columns
    self._est_query_sig2 = np.mean(np.abs(self._xp[:n_nonzero])) ** 2 if self.query_importance_sampling else np.mean(self._xp[:n_nonzero] ** 2)

    if self.verbose and self.query_importance_sampling:
      print(f'Query weights:\n{query_weights}')
      print(f'Combined weights:\n{self._atom_weights * query_weights}')
      print(f'Updated permutation:\n{self._permutation}')
  
  def exact_values(self, arms: np.ndarray) -> np.ndarray:
    """
    Compute the exact value for the specified arms, save this value, and return
    it.

    @param arms: The arms for which to compute the exact value
    @return: The exact values of the specified arms
    """
    assert self._x is not None, 'Query vector not set'

    if np.any(self.it[arms] < self.max_pulls):
      A_arms = self._A[arms, self._permutation] if self._Ap is None else self._Ap[arms]
      self._estimates[arms] = (A_arms @ self._xp) * self.temperature
      self._it[arms] = self.max_pulls
      self._var[arms] = 0
    
    return self._estimates[arms]
  
  def pull_arm(self, arm: int, it: int) -> float:
    """
    Pull an arm the given number of times and return its estimated value.

    @param arm: The arm to pull
    @param it: The number of times to pull the arm
    @return: The updated estimated value of the arm
    """
    assert self._x is not None, 'Query vector not set'
    
    return self.batch_pull(np.atleast_1d(arm), it)[0]

  def pull(self, arms: np.ndarray, its: np.ndarray) -> np.ndarray:
    """
    Pull the specified arms the provided number of times (may be distinct) and
    return the arms' updated estimated values.

    @param arms: The arms to pull
    @param its: The number of times to pull each arm
    @return: The updated estimated values of the specified arms
    """
    assert self._x is not None, 'Query vector not set'
    assert arms.size == its.size, 'The number of arms and pulls must be the same'

    for i in np.nonzero(its > self._it[arms])[0]:
      self.batch_pull(np.atleast_1d(arms[i]), its[i])
    
    return self._estimates[arms]
  
  def pull_to_var(
      self,
      arms: np.ndarray,
      var_threshold: Union[float, np.ndarray],
      init_pulls: int = DEFAULT_VAR_PULL_INIT,
      pull_mult: float = DEFAULT_VAR_PULL_INCR,
      fudge_factor_var: float = 1.0,
      batched: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pull the specified arms until the estimated variance of the mean is below
    the provided threshold.

    @param arms: The arms to pull
    @param var_threshold: The variance threshold(s)
    @param init_pulls: The initial number of pulls for each arm (default 16)
    @param pull_mult: The factor by which to increase the number of pulls
      (default 2.0)
    @param fudge_factor_var: The fudge factor for the variance of the mean
      (default 1.0)
    @param batched: The flag to enable batched pulls, should only be enabled if 
      all arms have been pulled the same number of times (default False)

    @return: The updated estimated values of the specified arms and the variance
      of the mean
    """
    assert self._x is not None, 'Query vector not set'

    threshold_var = var_threshold / fudge_factor_var
    to_pull = self._var[arms] > threshold_var
    num_pulls = init_pulls

    while np.any(to_pull):
      num_pulls_rounded = int(ceil(num_pulls))
      pulling = arms[np.nonzero(to_pull)[0]]
      if batched:
        self.batch_pull(pulling, num_pulls_rounded)
      else:
        self.pull(pulling, np.full(pulling.size, num_pulls_rounded))
      to_pull &= self._var[arms] > threshold_var
      num_pulls = min(self.max_pulls, num_pulls * pull_mult)

    return self._estimates[arms], self._var[arms] * fudge_factor_var

  def batch_pull(self, arms: np.ndarray, it: int) -> np.ndarray:
    """
    Pull the specified arms the given number of times and return their updated
    estimated values.

    The updated estimated value is based on the first it columns of the
    permutation. All arms must have been pulled the same amount in order to use
    batching. If the number of pulls exceeds the maximum number of pulls, or the
    arms have already been pulled it many times, the estimated value is not
    updated. If importance sampling is enabled, the estimated value is re-
    weighted accordingly.

    @param arms: The arms to pull
    @param it: The number of times to pull all arms
    @return: The updated estimated values of the specified arms
    """
    assert self._x is not None, 'Query vector not set'
    assert np.unique(self._it[arms]).size <= 1, 'All arms must have been pulled the same number of times'

    if self.verbose:
      print(f"Pulling arm(s):\n{arms}")
      print(f'Using {(it / self.max_pulls) * 100:.2f}% of the budget')

    if arms.size == 0 or it <= self._it[arms][0]:
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
      self._var[arms] = np.sum((A * self._xp[:next_it] * self.temperature) ** 2 * (1 - weights) / (weights ** 2), axis=1)

    # no importance sampling (equal weighting)
    else:
      self._estimates[arms] *= prev_it
      self._estimates[arms] += (self._Ap[arms, prev_it:next_it] @ self._xp[prev_it:next_it]) * (self.max_pulls * self.temperature)
      self._estimates[arms] /= next_it
      self._var[arms] = self.variance / next_it

    if next_it == self.max_pulls:
      self._var[arms] = 0

    self._it[arms] = next_it

    return self._estimates[arms]
