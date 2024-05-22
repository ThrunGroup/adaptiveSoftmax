from experiments.run_llm import run_llm
from experiments.run_mnl import run_mnl
from experiments.run_synthetic import run_and_plot_synthetic
from experiments.plotter import print_results

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
from mnl.mnl_constants import (
  MNIST,
  EUROSAT,
)

if __name__ == '__main__':
  # Run LLM experiments
  print("Running LLM experiments")
  for dataset in [WIKITEXT_DATASET, PENN_TREEBANK_DATASET]:
    for model in [GPT2, LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
      print(f"Running model {model} on dataset {dataset}")
      path = run_llm(
        dataset,
        model,
        delta=DELTA, 
        eps=EPS,
      )
      print_results(path)
      print()

  # Run MNL experiments
  print("Running MNL experiments")
  for dataset in [EUROSAT, MNIST]:
    print(f"Running MNL on dataset {dataset}")
    path = run_mnl(
      dataset,
      delta=DELTA, 
      eps=EPS,
    )
    print_results(path)
    print()

  # Run and plot synthetic data experiments
  plot_paths = run_and_plot_synthetic(n=100)
  print(f"Plots for synthetic data saved to {plot_paths}")


  