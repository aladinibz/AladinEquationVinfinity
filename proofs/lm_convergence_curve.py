#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Simulated LM convergence (χ² reduction over iterations)
iters = np.arange(1, 21)
chi2 = 100 * np.exp(-0.5 * (iters - 1)) + 5 * np.random.randn(20)

plt.figure(figsize=(12,8),dpi=1200)
plt.semilogy(iters, chi2, 'gold', lw=5, marker='o')
plt.title("Levenberg-Marquardt Convergence\nχ² Reduction over Iterations", fontsize=18)
plt.xlabel("Iteration", fontsize=14); plt.ylabel("χ² (log scale)", fontsize=14)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/lm_convergence_curve.png", dpi=1200)
plt.show()
