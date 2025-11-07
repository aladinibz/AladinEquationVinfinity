import numpy as np
import matplotlib.pyplot as plt

G = 4.3e-3
r = np.logspace(0, 2, 100)
v = 220 * np.ones_like(r)

a0 = 1.2e-8
g_N = G * 1e11 / r**2
v_aladin = np.sqrt(r * np.sqrt(g_N**2 + a0 * g_N))

plt.figure(figsize=(7,5))
plt.plot(r, v, 'k--', label='Observed')
plt.plot(r, v_aladin, 'blue', label='Aladin v∞')
plt.xlabel('r (kpc)')
plt.ylabel('v (km/s)')
plt.title('Aladin v∞ Rotation Curve — Fixed')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rotation_fix.png', dpi=300)
plt.close()
