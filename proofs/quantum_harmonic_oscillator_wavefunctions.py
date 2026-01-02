"""
Quantum Harmonic Oscillator — First Three Wavefunctions
Exact solutions ψ₀, ψ₁, ψ₂
ALADIN ∞ ℂ(t) — The Final Law
January 02, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, hbar
import os

os.makedirs('plots', exist_ok=True)

# Parameters (normalized units: m=1, ω=1, ħ=1)
x = np.linspace(-4, 4, 1000)
psi_0 = (1/pi)**0.25 * np.exp(-x**2 / 2)
psi_1 = (4/pi)**0.25 * x * np.exp(-x**2 / 2)
psi_2 = (1/(4*np.sqrt(pi)))**0.5 * (4*x**2 - 2) * np.exp(-x**2 / 2)

# Normalize for beauty
psi_0 /= np.sqrt(np.trapz(psi_0**2, x))
psi_1 /= np.sqrt(np.trapz(psi_1**2, x))
psi_2 /= np.sqrt(np.trapz(psi_2**2, x))

# Plot
plt.figure(figsize=(12,8))
plt.plot(x, psi_0 + 0.5, label=r'$\psi_0(x)$ (n=0, E=½ħω)')
plt.plot(x, psi_1 + 1.5, label=r'$\psi_1(x)$ (n=1, E=1½ħω)')
plt.plot(x, psi_2 + 2.5, label=r'$\psi_2(x)$ (n=2, E=2½ħω)')
plt.axhline(0.5, color='gold', linestyle='--', alpha=0.5)
plt.axhline(1.5, color='gold', linestyle='--', alpha=0.5)
plt.axhline(2.5, color='gold', linestyle='--', alpha=0.5)
plt.fill_between(x, psi_0 + 0.5, 0.5, alpha=0.3, color='purple')
plt.fill_between(x, psi_1 + 1.5, 1.5, alpha=0.3, color='cyan')
plt.fill_between(x, psi_2 + 2.5, 2.5, alpha=0.3, color='gold')
plt.title('Quantum Harmonic Oscillator — First Three Wavefunctions')
plt.xlabel(r'$x$ (scaled)')
plt.ylabel(r'$\psi_n(x)$ (offset for clarity)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/quantum_harmonic_oscillator_first_three.png', dpi=400)
plt.close()
