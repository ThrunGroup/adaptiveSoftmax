import numpy as np
import os
from llms.llm_constants import NUM_QUERY, WIKITEXT_DATASET, MISTRAL_7B
from llms.llm_utils import load_llm_data
from matplotlib import pyplot as plt
from tqdm import tqdm

def generate_weighted_permutation(weights: np.ndarray, gen=np.random.default_rng(0)):
  assert np.all(weights >= 0)
  with np.errstate(divide='ignore'):
    logits = np.log(weights) - np.log(np.sum(weights))
    perturbed_logits = logits + gen.gumbel(size=logits.size)
    permutation = perturbed_logits.argsort()[::-1]
  return permutation, logits, perturbed_logits

def run_and_plot_variances():
  np.random.seed(42)

  A, X = load_llm_data(WIKITEXT_DATASET, MISTRAL_7B, NUM_QUERY, testing=False)

  a = A[0]
  x = X[0]

  _, d = A.shape

  q = np.abs(x)/np.sum(np.abs(x))
  nonzero = q != 0
  n_nonzero = np.sum(nonzero != 0)

  sig2 = d**2 * np.var(a * x, ddof=1)
  sig2_imp = np.var((a * x / q)[nonzero], ddof=1)

  num_it = 20
  sample_sizes = np.linspace(10, d, 100, dtype=int)
  true = np.dot(a, x)
  lst_err_wr = []
  lst_err_wor = []
  lst_err_iwr = []
  lst_err_iwor = []
  lst_var_true = []
  lst_var_true_est = []
  gen = np.random.default_rng(0)

  for ss in tqdm(sample_sizes):
    err_wr = 0.0
    err_wor = 0.0
    err_iwr = 0.0
    err_iwor = 0.0

    true_var = 0.0
    true_est_var = 0.0

    for _ in range(num_it):
      sample_wr = np.random.choice(d, ss, replace=True)
      sample_wor = np.random.choice(d, ss, replace=False)
      sample_iwr = np.random.choice(d, ss, p=q, replace=True)
      permutation, logits, perturbed_logits = generate_weighted_permutation(q, gen)
      sample_iwor = permutation[:ss]

      est_wr = np.mean(a[sample_wr] * x[sample_wr]) * d
      est_wor = np.mean(a[sample_wor] * x[sample_wor]) * d
      est_iwr = np.mean(a[sample_iwr] * x[sample_iwr] / q[sample_iwr])
      threshold = -np.inf if ss == d else perturbed_logits[permutation[ss]]
      weights = 1 - np.exp(-np.exp(logits - threshold))
      weights = np.nan_to_num(weights, nan=1.0)
      est_iwor = np.sum((a * x / weights)[sample_iwor])

      err_wr += (est_wr - true) ** 2
      err_wor += (est_wor - true) ** 2
      err_iwr += (est_iwr - true) ** 2
      err_iwor += (est_iwor - true) ** 2

      pop = a * x
      true_var += np.sum(((pop ** 2) * ((1 - weights) / weights))[nonzero])
      true_est_var += np.sum(((pop ** 2) * (1 - weights) / (weights ** 2))[sample_iwor])

    err_wr /= num_it
    err_wor /= num_it
    err_iwr /= num_it
    err_iwor /= num_it
    true_var /= num_it
    true_est_var /= num_it

    if err_wr < 1e-6:
      err_wr = 0
    if err_wor < 1e-6:
      err_wor = 0
    if err_iwr < 1e-6:
      err_iwr = 0
    if err_iwor < 1e-6:
      err_iwor = 0
    if true_var < 1e-6:
      true_var = 0
    if true_est_var < 1e-6:
      true_est_var = 0

    lst_err_wr.append(err_wr)
    lst_err_wor.append(err_wor)
    lst_err_iwr.append(err_iwr)
    lst_err_iwor.append(err_iwor)
    lst_var_true.append(true_var)
    lst_var_true_est.append(true_est_var)

  bound = sig2 / sample_sizes
  bound_fpc = bound * np.sqrt((d - sample_sizes) / (d - 1))
  bound_imp = sig2_imp / sample_sizes
  bound_imp_fpc_sparse = bound_imp * np.maximum(0, (n_nonzero - sample_sizes)) / (n_nonzero - 1)

  plt.plot(sample_sizes, bound, label='Variance', linestyle='--', color=(0.6, 0.2, 0))
  plt.plot(sample_sizes, lst_err_wr, label='MSE [wr]', color=(0.6, 0.2, 0))

  plt.plot(sample_sizes, bound_imp, label='Variance [imp]', linestyle='--', color=(0.4, 0.4, 0))
  plt.plot(sample_sizes, lst_err_iwr, label='MSE [imp] [wr]', color=(0.4, 0.4, 0))

  plt.plot(sample_sizes, bound_imp_fpc_sparse, label='Variance [imp] [fpc-sparse]]', linestyle='--', color=(0.2, 0.6, 0))

  plt.plot(sample_sizes, lst_err_iwor, label='MSE [imp] [wor]', color=(0, 0.8, 0))
  plt.plot(sample_sizes, lst_var_true_est, label='New Variance Est.', linestyle='--', color=(0, 0.8, 0))

  plt.legend()
  plt.yscale('log')
  plt.xlabel('Sample Size')
  plt.ylabel('Error of Sample Mean')

  path = 'experiments/var_results/variances.pdf'
  os.makedirs(os.path.dirname(path), exist_ok=True)
  plt.savefig(path, format='pdf', bbox_inches='tight')
  return path

if __name__ == '__main__':
  run_and_plot_variances()