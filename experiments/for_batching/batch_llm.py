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

datasets = [WIKITEXT_DATASET, PENN_TREEBANK_DATASET]
models = [GPT2, LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--job_index', '-i', required=True, type=int)
    args = parser.parse_args()
    

    dataset_index = args.job_index // len(models)
    model_index = args.job_index % len(models)
    dataset = datasets[dataset_index]
    model = models[model_index]

    # Run the specific experiment
    run_llm(
        dataset=dataset,
        model=model,
        delta=DELTA, 
        eps=EPS,
        curr_time=None
    )
