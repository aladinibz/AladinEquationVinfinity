#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Your measured Δf and Q values from subjects 17/23/31
delta_f = [8.5e-10, 9.2e-10, 7.8e-10]  # linewidth (Hz)
q_vals = [5.06e10, 4.67e10, 5.51e10]  # Q = f / Δf

subjects = ['17 (5-MeO)', '23 (DMT)', '31 (Med)']

plt.figure(figsize=(12,8),dpi=1200)
plt.bar(subjects, q_vals, color='gold')
plt.ylabel("Q Factor")
plt.title("Lorentzian Fit Comparison Across Subjects\nQ at 43 Hz Resonance",fontsize=18)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/lorentzian_fit_comparison.png",dpi=1200)
