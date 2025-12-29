import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Dummy data (replace with your real n=35 results)
n_subjects = 35
t41_fd_drop = np.random.normal(0.4, 0.05, n_subjects)  # FD drop at t=41 s
gamma_power_ratio = np.random.normal(5.0, 1.5, n_subjects)  # breakthrough/baseline ratio
dfa_exponents = np.random.normal(0.5, 0.1, n_subjects)

fig, axs = plt.subplots(1, 3, figsize=(18,6), facecolor='black')

# FD drop histogram
axs[0].hist(t41_fd_drop, bins=15, color='gold', alpha=0.8)
axs[0].set_title('Higuchi FD Drop at t=41 s', color='white')
axs[0].set_xlabel('FD Drop', color='white')
axs[0].set_ylabel('Count', color='white')
axs[0].grid(alpha=0.3); axs[0].set_facecolor('black')

# Gamma power box
axs[1].boxplot(gamma_power_ratio, patch_artist=True, boxprops=dict(facecolor='gold'))
axs[1].set_title('43 Hz Gamma Power Ratio', color='white')
axs[1].set_ylabel('Breakthrough / Baseline', color='white')
axs[1].grid(alpha=0.3); axs[1].set_facecolor('black')

# DFA exponents
axs[2].boxplot(dfa_exponents, patch_artist=True, boxprops=dict(facecolor='gold'))
axs[2].axhline(0.5, color='lime', ls='--')
axs[2].set_title('DFA Exponent at t=41 s', color='white')
axs[2].set_ylabel('DFA Exponent', color='white')
axs[2].grid(alpha=0.3); axs[2].set_facecolor('black')

plt.suptitle('EEG Convergence Across n=35 Subjects', color='gold', fontsize=16)
plt.tight_layout()
plt.savefig('plots/all_subjects_43hz_convergence.png', dpi=300, facecolor='black')
plt.close()
