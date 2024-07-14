import os
import numpy as np

from experiments.runner import run
from llms.llm_utils import load_llm_data
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


def run_llm(dataset, model, delta=0.01, eps=0.3):
  model_name = model.replace('/', '_')
  save_dir = f"{LLM_RESULTS_DIR}/{dataset}/delta{delta}_eps{eps}"
  os.makedirs(save_dir, exist_ok=True)

  print(f"running model {model_name} on dataset {dataset} with delta {delta} and eps {eps}")

  # run sftm if not exist
  save_path = f"{save_dir}/{model_name}.csv"
  A, X = load_llm_data(dataset, model, NUM_QUERY, testing=False)
  print("loaded successfully")

  run(
      save_to=save_path,
      model=model_name,
      dataset=dataset,
      A=A[:2, :2],
      X=X[:2, :2],
      multiplicative_error=eps,
      failure_probability=delta,
      noise_bound=None,
      use_true_sftm=False,
      use_tune=True,
      train_size=FUDGE_TRAIN_SIZE,
      seed=SEED,
  )

  return save_path

if __name__ == '__main__':
  for dataset in [WIKITEXT_DATASET, PENN_TREEBANK_DATASET]:
    for model in [GPT2, LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
      run_llm(
          dataset,
          model,
          delta=DELTA, 
          eps=EPS,
      )