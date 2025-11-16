import numpy as np
import matplotlib.pyplot as plt

# --- Z-PINCH FROM ALADIN ---
r = np.logspace(18, 25, 100)
J = 1e18
B = 1e-6
rho = 1e-24
c = 3e8
F = 0.1 * (J * B) / (c * rho * r)

plt.loglog(r, F, 'gold', lw=3)
plt.axvline(1e21, color='cyan', ls='--', label='Galaxy Scale')
plt.xlabel('Radius r (m)')
plt.ylabel('F_Z-Pinch (N/m³)')
plt.title('ALADIN ∞ ℂ(t) — Z-Pinch from Equation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/z_pinch_aladin.png', dpi=300)
plt.show()
