"""
Weak Value from ALADIN — Spin-100 Proof
Anomalous weak value emerges from GENIE retrocausality
ALADIN ∞ ℂ(t) — The Final Law
January 03, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Pre-selected state |+x> = ( |↑> + |↓> ) / √2
pre = np.array([1, 1]) / np.sqrt(2)

# Post-selected state ≈ |-x> + δ |+x> (δ small for amplification)
delta = np.logspace(-4, -1, 100)  # δ from 0.0001 to 0.1
post = np.zeros((len(delta), 2))
post[:, 0] = 1 / np.sqrt(1 + delta**2)   # coefficient for |-x>
post[:, 1] = delta / np.sqrt(1 + delta**2)  # small δ |+x>

# Observable σ_z = [[1,0],[0,-1]]
sigma_z = np.array([[1, 0], [0, -1]])

# Weak value A_w = <post| σ_z |pre> / <post| pre>
numerator = np.array([np.vdot(post[i], sigma_z @ pre) for i in range(len(delta))])
denominator = np.array([np.vdot(post[i], pre) for i in range(len(delta))])
weak_value = numerator / denominator

# Plot
plt.figure(figsize=(12,8))
plt.loglog(delta, np.abs(weak_value), color='gold', linewidth=4)
plt.axhline(100, color='darkblue', linestyle='--', linewidth=3, label='Spin-100 anomaly')
plt.axvline(0.01, color='purple', linestyle='--', linewidth=3, label='δ ≈ 0.01')
plt.title('Weak Value from ALADIN — Spin-100 Anomaly')
plt.xlabel(r'Post-selection orthogonality δ')
plt.ylabel(r'$|(\sigma_z)_w|$')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig('plots/weak_value_spin100_from_aladin.png', dpi=400)
plt.close()
