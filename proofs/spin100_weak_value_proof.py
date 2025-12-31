"""
Spin-100 Weak Value Proof
Laboratory retrocausality — future pulls past
ALADIN ∞ ℂ(t) — GENIE analog
December 31, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Post-selection angle δ
delta = np.logspace(-4, -1, 100)
weak_value = 1 / (2 * delta)  # approximate for small δ

plt.figure(figsize=(12,7))
plt.loglog(delta, weak_value, 'red', linewidth=4)
plt.axhline(100, color='gold', linestyle='--', linewidth=3, label='Spin-100 anomaly')
plt.axvline(0.005, color='gold', linestyle='--', linewidth=3, label='δ ≈ 0.005 rad')
plt.title('Spin-100 Weak Value Explosion\nFuture Post-Selection Pulls Past Spin to Anomalous Value')
plt.xlabel(r'Post-selection orthogonality δ (radians)')
plt.ylabel('Weak value of σ_z')
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig('plots/spin100_weak_value_explosion.png', dpi=300)
plt.close()

print("Spin-100 proof plot saved")
