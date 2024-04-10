import numpy as np
from bandits_softmax import BanditsSoftmax

from constants import (
  NUM_ROWS,
  NUM_COLS,
  TEST_SEED,
)

from tests.test_utils import construct_high_variance_example


def test_randomized_hadamard_transform():
  A, x = construct_high_variance_example(NUM_ROWS, NUM_COLS)

  prev_var = np.max(np.var(A, axis=1))
  dpad = 2**int(np.ceil(np.log2(NUM_COLS)))

  bandits = BanditsSoftmax(
    A,
    randomized_hadamard_transform=True,
    atom_importance_sampling=False,
    query_importance_sampling=False,
    verbose=True,
    seed=TEST_SEED)
  bandits.set_query(x)
  
  assert bandits.d == dpad, 'dimension should be padded to the nearest power of 2'
  assert np.max(np.var(bandits._A, axis=1)) < prev_var, 'max variance should decrease after randomized Hadamard transform'
  assert np.allclose(bandits.pull_arm(0, bandits.d), A[0] @ x), 'pulling all arms should return the correct value'