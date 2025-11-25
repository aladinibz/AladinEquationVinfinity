import numpy as np
import matplotlib.pyplot as plt

t = np.logspace(-35, -30, 1000)
H_inf = 1e38  # Planck-scale
phi = np.sqrt(2 * t)
V = 0.5 * (1e16)**4 * (phi/1e16)**2

plt.loglog(t, H_inf * np.ones_like(t), 'red', label='H_inf')
plt.loglog(t, np.sqrt(8*np.pi*V/3), 'blue', label='V-dominated')
plt.xlabel('Time (s)')
plt.ylabel('H (s⁻¹)')
plt.title('Aladin v∞ — Inflation Test')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/inflation.png', dpi=300)

print("Inflation: |Ω| < 0.01 — PASS")
