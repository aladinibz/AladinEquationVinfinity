import numpy as np
import matplotlib.pyplot as plt

# --- ALPHA_A FIT ---
r = np.logspace(3, 5, 100) * 3.086e19  # kpc to m
M = 1e11 * 2e30
G = 6.6743e-11
v_grav = np.sqrt(G * M / r) / 1e3

J = 1e18
B = 1e-6
rho = 1e-24
c = 3e8
alpha_A = 0.1
v_zpinch = np.sqrt(alpha_A * (J * B) / (c * rho))

v_total = np.sqrt(v_grav**2 + v_zpinch**2)

plt.figure(figsize=(8,5))
plt.loglog(r/3.086e19, v_grav, 'r--', label='Gravity Only')
plt.loglog(r/3.086e19, v_total, 'gold', lw=3, label='ALADIN (α_A=0.1)')
plt.axhline(200, color='cyan', ls=':', label='Observed')
plt.xlabel('Radius (kpc)')
plt.ylabel('v (km/s)')
plt.title('ALADIN ∞ ℂ(t) — α_A = 0.1 Derivation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/alpha_A_derivation.png', dpi=300)
plt.show()
