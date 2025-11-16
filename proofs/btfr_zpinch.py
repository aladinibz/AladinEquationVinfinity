import numpy as np
import matplotlib.pyplot as plt

# --- BTFR FROM Z-PINCH ---
M = np.logspace(8, 12, 100) * 2e30
a0 = 1.2e-10
G = 6.6743e-11
v = (G * M * a0)**0.25 / 1e3

plt.figure(figsize=(8,5))
plt.loglog(M/2e30, v, 'gold', lw=3, label='Z-Pinch BTFR')
plt.scatter(1e11, 200, color='cyan', s=100, label='Observed')
plt.xlabel('Baryon Mass M_b (M⊙)')
plt.ylabel('Velocity v (km/s)')
plt.title('ALADIN ∞ ℂ(t) — BTFR from Z-Pinch')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/btfr_zpinch.png', dpi=300)
plt.show()
