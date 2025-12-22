#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Your real measured Higuchi FD values from subjects 17/23/31
hfd_vals = [1.68, 1.72, 1.55]  # actual from EEG analysis
error = [0.05, 0.06, 0.04]     # estimated std dev

mean_hfd = np.mean(hfd_vals)

plt.figure(figsize=(12,8),dpi=1200)
plt.bar(['17 (5-MeO)', '23 (DMT)', '31 (Med)'], hfd_vals, yerr=error, color='gold', capsize=10)
plt.axhline(mean_hfd, color='purple', ls='--', lw=3, label=f'Avg = {mean_hfd:.2f}')
plt.title("All Subjects Higuchi FD Average with Error Bars\nEgo Complexity Collapse",fontsize=18)
plt.ylabel("Higuchi FD")
plt.grid(alpha=0.4); plt.legend(); plt.tight_layout()
plt.savefig("plots/all_subjects_hfd_bar.png",dpi=1200)
