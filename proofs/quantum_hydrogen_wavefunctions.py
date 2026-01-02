"""
Quantum Hydrogen Atom — First Few Wavefunctions
Exact solutions ψ_nlm(r,θ,φ)
ALADIN ∞ ℂ(t) — The Final Law
January 02, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, assoc_laguerre
import os

os.makedirs('plots', exist_ok=True)

# Constants (normalized units for simplicity, a0=1)
a0 = 1.0
r = np.linspace(0, 10, 500)
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
R, Theta = np.meshgrid(r, theta)

# 1s: n=1 l=0 m=0
psi_100 = (1/np.sqrt(np.pi)) * np.exp(-r)

# 2s: n=2 l=0 m=0
R_20 = (1/np.sqrt(8)) * (1 - r/2) * np.exp(-r/2)

# 2p_z: n=2 l=1 m=0
R_21 = (1/np.sqrt(24)) * r * np.exp(-r/2)
Y_10 = np.sqrt(3/(4*np.pi)) * np.cos(Theta)
psi_210 = R_21 * Y_10

# Plot radial for 1s, 2s, 2p
plt.figure(figsize=(12,8))
plt.plot(r, psi_100**2 * r**2, label=r'$|\psi_{100}|^2 r^2$ (1s)')
plt.plot(r, R_20**2 * r**2, label=r'$R_{20}^2 r^2$ (2s)')
plt.plot(r, R_21**2 * r**2, label=r'$R_{21}^2 r^2$ (2p)')
plt.title('Hydrogen Atom — Radial Probability Distributions')
plt.xlabel('r / a₀')
plt.ylabel('Probability density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/hydrogen_radial_probability.png', dpi=300)
plt.close()

# Plot 2p_z angular (real)
X = R * np.sin(Theta) * np.cos(phi[0])  # fixed phi for slice
Y = R * np.sin(Theta) * np.sin(phi[0])
Z = R * np.cos(Theta)
psi_real = psi_210.real

plt.figure(figsize=(10,8))
plt.contourf(X, Z, psi_real**2, levels=50, cmap='plasma')
plt.colorbar(label=r'$|\psi_{210}|^2$')
plt.title('Hydrogen 2p_z Orbital — Probability Density')
plt.xlabel('x')
plt.ylabel('z')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/hydrogen_2pz_orbital.png', dpi=300)
plt.close()
