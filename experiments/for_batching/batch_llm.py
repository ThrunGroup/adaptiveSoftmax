
import time
from argparse import ArgumentParser
from experiments.run_llm import run_llm

from llms.llm_constants import (
  WIKITEXT_DATASET,
  PENN_TREEBANK_DATASET,

  GPT2, 
  GEMMA_7B,
  MISTRAL_7B,
  LLAMA_3_8B,

  DELTA, 
  EPS,
)

deltas = [0.01, 0.05, 0.1]
datasets = [WIKITEXT_DATASET, PENN_TREEBANK_DATASET]
models = [GPT2, LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--job_index', '-i', required=True, type=int)
    args = parser.parse_args()
    
    # get the correct mappings
    job_i = args.job_index
    dataset_i = job_i // (len(models) * len(deltas))
    model_i = (job_i % (len(models) * len(deltas))) // len(deltas)
    delta_i = job_i % len(deltas)

    dataset = datasets[dataset_i]
    model = models[model_i]
    delta = deltas[delta_i]

    print(dataset, model, delta)

    # Run the specific experiment
    run_llm(
        dataset=dataset,
        model=model,
        delta=delta, 
        eps=EPS,
    )
