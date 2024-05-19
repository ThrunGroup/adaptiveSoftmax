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

def plot_scaling(dimensions, naive_budgets, budgets, success_rates, save_to):
    plt.figure(figsize=(10, 6))
    
  
    plt.plot(dimensions, naive_budgets, 's-', color='red', label='Naive Budgets')
    plt.plot(dimensions, budgets, 'o-', color='blue', label='Budgets')
    plt.yscale('log')
    
    plt.xlabel('Dimension (d)')
    plt.ylabel('Budget')
    plt.title(f'Budget vs Dimension\n Success rates = {success_rates}')
    
    plt.legend()
    plt.savefig(f"{save_to}_scaling_plots.png")
    plt.close()


if __name__ == '__main__':
  data = pd.read_csv('experiments/llm_results/delta_0.01/0.005_gpt2.csv')
  data = clean_singleton_np_array_columns(data)
  
  print(get_budget_and_success_rate(data))