#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

iters = np.arange(1, 21)
delta = 1.0 * np.exp(-0.3 * (iters - 1)) + 0.1 * np.random.randn(20)  # typical decay

plt.figure(figsize=(12,8),dpi=1200)
plt.semilogy(iters, delta, 'gold', lw=5, marker='o')
plt.title("Trust Radius Evolution in TRF\nOver Iterations",fontsize=18)
plt.xlabel("Iteration",fontsize=14); plt.ylabel("Trust Radius Δ (log scale)",fontsize=14)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/trust_radius_evolution.png",dpi=1200)
