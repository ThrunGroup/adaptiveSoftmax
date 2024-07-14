import os
import numpy as np
import pandas as pd
import argparse

from typing import Tuple
from tqdm import tqdm

from adaptive_softmax.sftm import SFTM
from llms.llm_constants import (
  GPT2, 
  GEMMA_7B,
  MISTRAL_7B,
  LLAMA_3_8B
)

DEFAULT_SEED = 42

# NOTE: recommend a train size of around 100 (too much more takes too long)
# NOTE: recommend a failure probability of 10% but we may want smaller (ask Tavor)
# NOTE: recommend a multiplicative error of 0.3

def split_train_test(
    X: np.ndarray,
    train_size: int,
    seed: int = DEFAULT_SEED,
    ) -> Tuple[np.ndarray, np.ndarray]:
  gen = np.random.default_rng(seed)
  idx = gen.permutation(X.shape[0])
  X_train = X[idx[:train_size]]
  X_test = X[idx[train_size:]]
  return X_train, X_test


def run_sftm(
    save_to: str,
    model: str,
    dataset: str,
    sftm: SFTM,
    X: np.ndarray,
    fudge_bandits: float = 1.0,
    fudge_log_norm: float = 1.0,
    seed: int = DEFAULT_SEED,
    quiet: bool = False,
    ) -> pd.DataFrame:
  results = []
  it = enumerate(X)
  if not quiet:
    it = tqdm(list(it))
  for i, x in it:
    best_arm_hat, p_hat, log_S_hat = sftm.adaptive_softmax(
      x, fudge_bandits=fudge_bandits, fudge_log_norm=fudge_log_norm)
    best_arm, p, log_S = sftm.softmax(x)
    total_budget = np.sum(sftm.bandits.it)

    res = {}

    res['model'] = model
    res['dataset'] = dataset
    res['n'] = sftm.n
    res['d'] = sftm.d
    res["density"] = np.count_nonzero(sftm.A)
    res['query'] = i
    res['d_not_sparse'] = sftm.bandits.max_pulls
    res['budget_total'] = total_budget

    res['best_arm_hat'] = best_arm_hat
    res['p_hat_best_arm_hat'] = p_hat
    res['log_S_hat'] = log_S_hat

    res['best_arm'] = best_arm
    res['p_best_arm'] = p
    res['log_S'] = log_S

    eps = sftm.multiplicative_error
    delta = sftm.failure_probability

    eps = eps if sftm.exact_pull_best_arm else eps / 4
    delta = delta / 2 if sftm.exact_pull_best_arm else delta / 3

    sftm.bandits.set_query(x, seed)
    sftm.log_norm_estimation(
      eps, delta, fudge_factor=fudge_log_norm, first_pull_batched=True)
    res['budget_log_norm_estimation'] = np.sum(sftm.bandits.it)

    sftm.bandits.set_query(x, seed)
    sftm.best_arms(
      delta, 1, fudge_factor=fudge_bandits)
    res['budget_best_arms'] = np.sum(sftm.bandits.it)

    if sftm.exact_pull_best_arm:
      res['budget_estimate_best_arm'] = sftm.bandits.max_pulls
    else:
      sftm.bandits.set_query(x, seed)
      sftm.estimate_arm_logits(best_arm_hat, eps, delta)
      res['budget_estimate_best_arm'] = np.sum(sftm.bandits.it)

    results.append(res)

  # don't overwrite
  data = pd.DataFrame(results)
  if save_to is not None:
    data.to_csv(save_to, mode='w', header=True, index=False)

  return data


def run(
    save_to: str,
    model: str,
    dataset: str,
    A: np.ndarray,
    X: np.ndarray,
    multiplicative_error: float, # eps
    failure_probability: float, # delta
    noise_bound: float, # sig2 NOTE can be None to estimate sig2 instead
    use_true_sftm: bool,
    use_tune: bool,
    train_size: int,
    seed: int = DEFAULT_SEED,
    quiet: bool = False,
    ) -> pd.DataFrame:
  
  sftm = None
  if use_true_sftm:
    sftm = SFTM(
      A,
      temperature=1.0,
      multiplicative_error=multiplicative_error,
      failure_probability=failure_probability,
      noise_bound=noise_bound,
      atom_importance_sampling=False,
      query_importance_sampling=False,
      randomized_hadamard_transform=False,
      exact_pull_best_arm=False,
      max_init_pull_budget=1,
      verbose=False,
      seed=seed,
    )
  else:
    sftm = SFTM(
      A,
      temperature=1.0,
      multiplicative_error=multiplicative_error,
      failure_probability=failure_probability,
      noise_bound=noise_bound,
      atom_importance_sampling=True,
      query_importance_sampling=True,
      randomized_hadamard_transform=False,
      exact_pull_best_arm=True,
      max_init_pull_budget=0.1,
      verbose=False,
      seed=seed,
    )

  fudge_bandits = 1.0
  fudge_log_norm = 1.0
  if use_tune:
    X_train, X = split_train_test(X, train_size, seed)
    fudge_bandits, fudge_log_norm = sftm.tune_fudge_factors(X_train, verbose=True)

  return run_sftm(save_to, model, dataset, sftm, X, fudge_bandits, fudge_log_norm, seed, quiet, is_sparse)