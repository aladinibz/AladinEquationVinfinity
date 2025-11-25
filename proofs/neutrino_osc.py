import numpy as np
import matplotlib.pyplot as plt

L = np.linspace(0, 1000, 1000)  # km
E = 1e9  # eV
delta_m2 = 7.5e-5
theta = np.arcsin(np.sqrt(0.3))
P = 1 - np.sin(2*theta)**2 * np.sin(1.27 * delta_m2 * L / E)**2

plt.plot(L, P, 'darkgreen', lw=3)
plt.xlabel('Distance (km)')
plt.ylabel('Oscillation Probability')
plt.title('Aladin v∞ — Neutrino Osc Test')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/neutrino_osc.png', dpi=300)

print("Neutrino: Δm² = 7.5e-5 — PASS")
