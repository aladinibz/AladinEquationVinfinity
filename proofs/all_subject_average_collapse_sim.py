import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Your real measured values (from subjects 17/23/31)
hfd_vals = [1.68, 1.72, 1.55]  # Higuchi FD
alpha_vals = [0.85, 0.88, 0.75]  # DFA α

mean_hfd = np.mean(hfd_vals)
mean_alpha = np.mean(alpha_vals)

plt.figure(figsize=(12,8),dpi=1200)
plt.bar(['Avg Higuchi FD', 'Avg DFA α'], [mean_hfd, mean_alpha], color=['gold', 'cyan'])
plt.title("ALADIN ∞ ℂ(t) — All Subjects Average Collapse\nEnlightenment Measured",fontsize=18)
plt.ylabel("Average Value")
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/all_subject_average_collapse.png",dpi=1200)
print("SAVED: plots/all_subject_average_collapse.png — check Files → refresh → download")
plt.show()
