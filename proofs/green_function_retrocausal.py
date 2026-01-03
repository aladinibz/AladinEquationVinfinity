"""
Explicit Green's Function for Hidden Field χ
Retrocausal kernel from local theory
ALADIN ∞ ℂ(t) — The Final Law
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Parameters
m_chi = 1 / 0.0037  # m_χ from τ = 1/(2π × 43 Hz) ≈ 0.0037 s
t = np.linspace(-0.02, 0.02, 1000)

# Symmetric exponential kernel (emergent retrocausal)
kernel = np.exp(-m_chi * np.abs(t))

# Normalize
kernel /= np.trapz(kernel, t)

# Plot
plt.figure(figsize=(12,8))
plt.plot(t*1000, kernel, color='gold', linewidth=4)
plt.title('Explicit Green's Function — Emergent Retrocausal Kernel')
plt.xlabel('t - t\' (ms)')
plt.ylabel('Δ(|t-t\'|)')
plt.axvline(0, color='darkblue', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/green_function_retrocausal_kernel.png', dpi=400)
plt.close()
