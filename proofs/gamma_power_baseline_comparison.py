import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Replace with your real data (e.g., load from CSV or compute from .bdf)
n = 35
baseline_gamma = np.random.normal(1.0, 0.2, n)  # baseline power
breakthrough_gamma = np.random.normal(5.0, 1.5, n)  # breakthrough power

# Ratio
gamma_ratio = breakthrough_gamma / baseline_gamma

plt.figure(figsize=(8,6),facecolor='black')
plt.boxplot(gamma_ratio, patch_artist=True, boxprops=dict(facecolor='gold'))
plt.axhline(1.0, color='white', ls='--', alpha=0.5)
plt.ylabel('Gamma Power Ratio (Breakthrough / Baseline)', color='white')
plt.title('43 Hz Gamma Power: Breakthrough vs Baseline (n=35)', color='gold')
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/gamma_power_baseline_comparison.png', dpi=300, facecolor='black')
plt.close()
