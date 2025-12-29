import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Replace with your real DFA α values from s01–s35 scripts
n = 35
alpha_pre = np.random.normal(1.0, 0.1, n)   # pre t=41 s
alpha_post = np.random.normal(0.5, 0.05, n) # post t=41 s

# Hurst H: H = α if α < 1, else H = α - 1
hurst_pre = np.where(alpha_pre < 1, alpha_pre, alpha_pre - 1)
hurst_post = np.where(alpha_post < 1, alpha_post, alpha_post - 1)

fig, axs = plt.subplots(1, 2, figsize=(14,6), facecolor='black')

# DFA α box plots
axs[0].boxplot([alpha_pre, alpha_post], labels=['Pre t=41 s', 'Post t=41 s'], patch_artist=True,
               boxprops=dict(facecolor='gold'))
axs[0].axhline(0.5, color='lime', ls='--')
axs[0].set_title('DFA Scaling Exponent α', color='white')
axs[0].set_ylabel('α', color='white')
axs[0].grid(alpha=0.3); axs[0].set_facecolor('black')

# Hurst H box plots
axs[1].boxplot([hurst_pre, hurst_post], labels=['Pre t=41 s', 'Post t=41 s'], patch_artist=True,
               boxprops=dict(facecolor='gold'))
axs[1].axhline(0.5, color='lime', ls='--')
axs[1].set_title('Hurst Exponent H', color='white')
axs[1].set_ylabel('H', color='white')
axs[1].grid(alpha=0.3); axs[1].set_facecolor('black')

plt.suptitle('DFA α vs Hurst H: n=35 Breakthroughs', color='gold', fontsize=16)
plt.tight_layout()
plt.savefig('plots/dfa_hurst_comparison.png', dpi=300, facecolor='black')
plt.close()
