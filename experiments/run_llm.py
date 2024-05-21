import os
import time
import numpy as np

from experiments.runner import run
from llms.llm_utils import load_llm_matrices
from llms.llm_constants import (
  WIKITEXT_DATASET,
  PENN_TREEBANK_DATASET,

  GPT2, 
  GEMMA_7B,
  MISTRAL_7B,
  LLAMA_3_8B,
  LLM_RESULTS_DIR,

  SEED,
  DELTA, 
  EPS,
  FUDGE_TRAIN_SIZE,
  NUM_QUERY,
)


def run_llm(curr_time, dataset, model, delta=0.01, eps=0.3):
  model_name = model.replace('/', '_')
  save_dir = f"{LLM_RESULTS_DIR}/{curr_time}/{dataset}/delta{delta}_eps{eps}"
  os.makedirs(save_dir, exist_ok=True)

  # run sftm if not exist
  save_path = f"{save_dir}/{model_name}.csv"
  if not any(os.scandir(save_dir)):
    A, X = load_llm_matrices(dataset, model, NUM_QUERY, testing=False)
    print("loaded successfully")
    run(
        save_to=f"{save_path}_raw.csv",
        model=model_name,
        dataset=dataset,
        A=A,
        X=X,
        multiplicative_error=eps,
        failure_probability=delta,
        noise_bound=None,
        use_true_sftm=False,
        use_tune=True,
        train_size=FUDGE_TRAIN_SIZE,
        seed=SEED,
    )

if __name__ == '__main__':
  curr_time = time.strftime("%H:%M:%S", time.gmtime())
  for dataset in [WIKITEXT_DATASET, PENN_TREEBANK_DATASET]:
    for model in [GPT2, LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
      run_llm(
          curr_time,
          dataset,
          model,
          delta=DELTA, 
          eps=EPS,
      )