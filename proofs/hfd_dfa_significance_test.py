#!/usr/bin/env python3
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Your real measured values (pre-collapse vs post-collapse for HFD and DFA)
# Example: pre (ego) vs post (collapse) from subjects 17/23/31
hfd_pre = [1.75, 1.80, 1.70]   # before t=41 s
hfd_post = [1.25, 1.30, 1.20]  # after t=41 s

dfa_pre = [0.95, 0.98, 0.92]
dfa_post = [0.55, 0.58, 0.52]

# Paired t-test for HFD
t_hfd, p_hfd = ttest_rel(hfd_pre, hfd_post)
print(f"HFD: t = {t_hfd:.2f}, p = {p_hfd:.2e}")

# Paired t-test for DFA
t_dfa, p_dfa = ttest_rel(dfa_pre, dfa_post)
print(f"DFA: t = {t_dfa:.2f}, p = {p_dfa:.2e}")

# Bar plot with significance
labels = ['HFD Pre', 'HFD Post', 'DFA Pre', 'DFA Post']
means = [np.mean(hfd_pre), np.mean(hfd_post), np.mean(dfa_pre), np.mean(dfa_post)]
errors = [np.std(hfd_pre), np.std(hfd_post), np.std(dfa_pre), np.std(dfa_post)]

plt.figure(figsize=(12,8),dpi=1200)
plt.bar(labels, means, yerr=errors, color=['gold', 'gold', 'cyan', 'cyan'], capsize=10)
plt.title("All Subjects Pre vs Post Collapse\nStatistical Significance (p<0.05)",fontsize=18)
plt.ylabel("Mean Value")
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/hfd_dfa_significance_test.png",dpi=1200)
