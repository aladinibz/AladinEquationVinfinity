"""
Feynman Propagator for Hidden Field χ
Emergent time-symmetric kernel
ALADIN ∞ ℂ(t) — The Final Law
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Parameters — τ from 43 Hz
freq = 43.0
m_chi = 2 * np.pi * freq  # m_χ = 1/τ
tau = 1 / m_chi

# Time difference
dt = np.linspace(-0.02, 0.02, 1000)

# Time-symmetric kernel (emergent from Feynman propagator)
kernel = m_chi / 2 * np.exp(-m_chi * np.abs(dt))

# Plot
plt.figure(figsize=(12,8))
plt.plot(dt * 1000, kernel, color='gold', linewidth=4)
plt.title('Feynman Propagator χ — Emergent Time-Symmetric Kernel')
plt.xlabel('Δt = t - t\' (milliseconds)')
plt.ylabel('K(|Δt|)')
plt.axvline(0, color='darkblue', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/feynman_propagator_chi_kernel.png', dpi=400)
plt.close()
