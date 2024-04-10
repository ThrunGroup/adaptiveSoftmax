import numpy as np
from bandits_softmax import BanditsSoftmax

def test_randomized_hadamard_transform():
  seed = 42
  n = 100
  d = 256

  gen = np.random.default_rng(seed)
  dpad = 2 ** int(np.ceil(np.log2(d)))
  A = gen.normal(size=(n, d))
  A[:, ::64] = 100 * gen.choice([-1, 1], size=(n, d // 64))
  x = np.ones(d)
  prev_var = np.max(np.var(A, axis=1))

  bandits = BanditsSoftmax(
    A,
    randomized_hadamard_transform=True,
    atom_importance_sampling=False,
    query_importance_sampling=False,
    verbose=True,
    seed=seed)
  bandits.set_query(x)
  
  assert bandits.d == dpad, 'dimension should be padded to the nearest power of 2'
  assert np.max(np.var(bandits._A, axis=1)) < prev_var, 'max variance should decrease after randomized Hadamard transform'
  assert np.allclose(bandits.pull_arm(13, bandits.d), A[13] @ x), 'pulling all arms should return the correct value'