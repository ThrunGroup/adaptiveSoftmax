import sys
import numpy as np
from experiments.runner import run

def get_params(job_id):
  """
  Assumes job_id is between [0-11]. 
  This models the 4 llms and 3 deltas we'll be testing
  """
  DELTAS = [0.01, 0.05, 0.1]
  MODELS = [
    "gpt2", 
    "meta-llama_Meta-Llama-3-8B",
    "mistralai_Mistral-7B-v0.1",
    "google_gemma-7b",
  ]  
  dataset = "wikitext_512"  # TODO: add penn_treebank (will be 24 tasks now)
  return MODELS[job_id % 4], DELTAS[job_id // 4], dataset


def run_llms(job_id):
  model, delta, dataset = get_params(job_id)
  path = f"testing_model_{model}_{dataset}.npz"

  A = np.load(f"llms/weights/{path}")
  X = np.load(f"llms/x_matrix/{path}")

  run(
    model=model,
    dataset=dataset,
    A=A,
    X=X,
    multiplicative_error=0.3,
    failure_probability=delta,
    noise_bound=None,
    use_true_sftm=False,
    use_tune=True,
    train_size=100,
    seed=42,
  )
  

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <job_id>")
        sys.exit(1)
    
    job_id = int(sys.argv[1])
    run_llms(job_id)