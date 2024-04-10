import numpy as np
from bandits_softmax import BanditsSoftmax

from constants import (
  NUM_ROWS,
  NUM_COLS,
  TEST_SEED,
)

from tests.test_utils import construct_high_variance_example

def test_importance_sampling(
    atom_importance_sampling: bool = True,
    query_importance_sampling: bool = True,
):
  d = 100
  A, x = construct_high_variance_example(NUM_ROWS, d)

  bandits = BanditsSoftmax(
    A,
    atom_importance_sampling=atom_importance_sampling,
    query_importance_sampling=query_importance_sampling,
    randomized_hadamard_transform=False,
    verbose=True,
    seed=TEST_SEED)
  bandits.set_query(x)

  bandits_no_importance = BanditsSoftmax(
    A,
    atom_importance_sampling=False,
    query_importance_sampling=False,
    randomized_hadamard_transform=False,
    verbose=False,
    seed=TEST_SEED)
  bandits_no_importance.set_query(x)

  true = A @ x

  errors = []
  errors_no_importance = []
  for i in range(1, d + 1):
    all_arms = np.arange(NUM_ROWS)
    errors.append(np.mean(np.abs(bandits.batch_pull(all_arms, i) - true)))
    errors_no_importance.append(np.mean(np.abs((bandits_no_importance.batch_pull(all_arms, i) - true))))

  average_error = sum(errors) / d
  average_error_no_importance = sum(errors_no_importance) / d

  print(f'Average error with importance sampling: {average_error}')
  print(f'Average error without importance sampling: {average_error_no_importance}')
  
  assert average_error < average_error_no_importance, 'importance sampling should reduce error'
  assert np.allclose(bandits.pull_arm(0, d), A[0] @ x), 'pulling all arms should return the correct value'

def test_additional_importance_sampling():
  test_importance_sampling(atom_importance_sampling=False, query_importance_sampling=True)
  test_importance_sampling(atom_importance_sampling=True, query_importance_sampling=False)

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