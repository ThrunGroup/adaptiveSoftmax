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
  FUDGE_TRAIN_SIZE,
  NUM_QUERY,
)


def run_llm(curr_time, dataset, model, delta=0.01, eps=0.3):
  print(model, dataset)
  model_name = model.replace('/', '_')
  save_dir = f"{LLM_RESULTS_DIR}/{curr_time}/{dataset}/delta{delta}_eps{eps}"
  os.makedirs(save_dir, exist_ok=True)

  # run sftm if not exist
  save_path = f"{save_dir}/{model_name}.csv"
  if not any(os.scandir(save_dir)):
    n_query = NUM_QUERY if dataset == WIKITEXT_DATASET else NUM_QUERY // 2  # TODO: change this once we get the 1000 query penn_treebank
    loading_path = f"{model_name}_{dataset}_query{n_query}.npz"  

    A = np.load(f"{LLM_WEIGHTS_DIR}/{loading_path}", allow_pickle=False)['data']
    X = np.load(f"{LLM_XS_DIR}/{loading_path}", allow_pickle=False)['data']

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