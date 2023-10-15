
import matplotlib.pyplot as plt


def plot_sigma_distribution(A, x, arm_index=0):
    num_bins = x.shape[0] ** 0.5
    elmul = A[arm_index] * x
    plt.hist(elmul, bins=int(num_bins))
