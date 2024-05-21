import os
import time
import numpy as np

from experiments.runner import run
from llms.llm_constants import (
  WIKITEXT_DATASET,
  PENN_TREEBANK_DATASET,

  GPT2, 
  GEMMA_7B,
  MISTRAL_7B,
  LLAMA_3_8B,

  LLM_WEIGHTS_DIR,
  LLM_XS_DIR,
  LLM_RESULTS_DIR,

  SEED,
  DELTA, 
  EPS,
  TRAIN_SIZE,
)


def run_llm(delta=0.01, eps=0.3):
  """
  Runs a total of 8 experiments (4 llms, 2 datasets) for provided delta and eps.
  """
  curr_time = time.strftime("%H:%M:%S", time.gmtime())
  for dataset in [WIKITEXT_DATASET, PENN_TREEBANK_DATASET]:
    for model in [GPT2, GEMMA_7B, MISTRAL_7B, LLAMA_3_8B]:
        model_name = model.replace('/', '_')

        loading_path = f"testing_{model_name}_{dataset}_512.npz"
        A = np.load(f"{LLM_WEIGHTS_DIR}/{loading_path}", allow_pickle=False)['data']
        X = np.load(f"{LLM_XS_DIR}/{loading_path}", allow_pickle=False)['data']

        save_dir = f"{LLM_RESULTS_DIR}_{curr_time}/{dataset}/delta{delta}_eps{eps}"
        os.makedirs(save_dir, exist_ok=True)
        run(
            save_to=f"{save_dir}/{model_name}.csv",
            model=model_name,
            dataset=dataset,
            A=A,
            X=X,
            multiplicative_error=eps,
            failure_probability=delta,
            noise_bound=None,
            use_true_sftm=False,
            use_tune=True,
            train_size=TRAIN_SIZE,
            seed=SEED,
        )

if __name__ == '__main__':
    run_llm(delta=DELTA, eps=EPS)