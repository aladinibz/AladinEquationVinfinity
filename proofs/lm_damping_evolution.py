#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Simulated LM damping λ evolution (typical behavior)
iters = np.arange(1, 21)
lambda_vals = [1e4 * np.exp(-0.5 * (iters - 1)) + 1e-2 * np.random.randn(20)]  # starts high, drops

plt.figure(figsize=(12,8),dpi=1200)
plt.semilogy(iters, lambda_vals, 'gold', lw=5, marker='o')
plt.title("Levenberg-Marquardt Damping Parameter λ Evolution\nOver Iterations",fontsize=18)
plt.xlabel("Iteration",fontsize=14); plt.ylabel("Damping λ (log scale)",fontsize=14)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/lm_damping_evolution.png",dpi=1200)
