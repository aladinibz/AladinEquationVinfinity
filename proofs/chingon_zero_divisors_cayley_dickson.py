"""
Chingon Zero-Divisors — Cayley-Dickson Construction
Illusion canceled — explorers free
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Cayley-Dickson steps
steps = np.arange(0, 8)
dims = 2**steps
labels = ['ℝ', 'ℂ', 'ℍ', '𝕆', 'Sedenions', 'Pathions', 'Chingon', 'Next']

# Zero-divisors start at sedenions (16D)
zero = np.zeros(8)
zero[4:] = [1, 100, 1e6, 1e12]  # illustrative explosion

plt.figure(figsize=(14,8))
plt.semilogy(steps, dims, 'o-', color='gold', linewidth=5, markersize=12, label='Dimension')
plt.semilogy(steps[4:], zero[4:], 's--', color='purple', linewidth=5, markersize=12, label='Zero-Divisors Explosion')
for i, label in enumerate(labels):
    plt.text(steps[i], dims[i]*2, label, ha='center', fontsize=14, fontweight='bold')
plt.title('Cayley-Dickson Construction — Zero-Divisors Emerge')
plt.xlabel('Doubling Generation')
plt.ylabel('Scale (log)')
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/chingon_zero_divisors_cayley_dickson.png', dpi=400)
plt.close()
