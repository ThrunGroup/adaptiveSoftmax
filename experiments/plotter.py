import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_singleton_np_array_columns(data: pd.DataFrame):
  for col in data.columns:
    data[col] = data[col].apply(lambda x: float(x.strip('[]')) if (isinstance(x, str) and '[' in x) else x)
  return data

def get_budget_and_success_rate(data: pd.DataFrame):
  budget = np.mean(data['budget_total'] / (data['d'] * data['n']))
  bandit_success = data['best_arm'] == data['best_arm_hat']
  log_norm_success = np.abs(data['p_hat_best_arm_hat'] - data['p_best_arm']) / data['p_best_arm'] <= 0.3
  success_rate = np.mean(bandit_success & log_norm_success)
  return budget, success_rate

def get_scaling_param(path_dir:str):
  dimensions = []
  budgets = []
  naive_budgets = []

  budget_percentages = []
  budget_stds = []
  success_rates = []

  for file in os.listdir(path_dir):
      if "plot" in file: continue
      data = pd.read_csv(os.path.join(path_dir, file))
      data = clean_singleton_np_array_columns(data)

      dimensions.append(int(np.mean(data['d'])))
      budgets.append(int(np.mean(data['budget_total'])))
      budget_stds.append(np.std(data['budget_total']))
      naive_budgets.append(int(np.mean(data['d'] * data['n'])))

      budget_percentage, success_rate = get_budget_and_success_rate(data)
      budget_percentages.append(budget_percentage)
      success_rates.append(success_rate)

  # Convert lists to numpy arrays for sorting
  dimensions = np.array(dimensions)
  budgets = np.array(budgets)
  naive_budgets = np.array(naive_budgets)
  budget_percentages = np.array(budget_percentages)
  success_rates = np.array(success_rates)
  budget_stds = np.array(budget_stds)

  sort_indices = np.argsort(naive_budgets)
  dimensions = dimensions[sort_indices]
  budgets = budgets[sort_indices]
  naive_budgets = naive_budgets[sort_indices]
  budget_percentages = budget_percentages[sort_indices]
  success_rates = success_rates[sort_indices]
  budget_stds = budget_stds[sort_indices]

  return dimensions.tolist(), budgets.tolist(), naive_budgets.tolist(), budget_stds, budget_percentages, success_rates.tolist()


def plot_scaling(dimensions, naive_budgets, budgets, stds, percentages, success_rates, save_to):
    plt.figure(figsize=(10, 6))
  
    # Plot budgets first
    print(stds)
    print(budgets)
    plt.plot(dimensions, naive_budgets, 's-', color='red', label='Naive Budgets')
    plt.errorbar(dimensions, budgets, yerr=stds, fmt='o-', color='blue', label='Budgets', capsize=5)
    plt.yscale('log')

    #values = np.log(np.log(np.array(dimensions)))
    #plt.plot(dimensions, budgets[0] * values / values[0] )

    plt.xlabel('Dimension (d)')
    plt.ylabel('Budget')
    plt.title(
    f"""
      Percentages = {", ".join(format(percentage, ".3f") for percentage in percentages)}
      Success rates = {", ".join(format(rate, ".2f") for rate in success_rates)}
    """)
    
    plt.legend()
    plt.savefig(f"{save_to}/scaling_plots.png")
    plt.close()

if __name__ == '__main__':
  data = pd.read_csv('experiments/llm_results/penn_treebank/delta_0.01/0.05_google_gemma-7b.csv')
  data = clean_singleton_np_array_columns(data)
  print("===>", get_budget_and_success_rate(data))
  