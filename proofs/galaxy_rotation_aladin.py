import numpy as np
import matplotlib.pyplot as plt

r = np.logspace(0, 1.5, 100)
v_obs = 220 * np.ones_like(r)
G = 4.3e-3
M = 1e11
a0 = 1.2e-10
g_N = G * M / r**2
v_aladin = np.sqrt(r * np.sqrt(g_N**2 + a0 * g_N))

plt.figure(figsize=(7,5))
plt.plot(r, v_obs, 'k--')
plt.plot(r, v_aladin, 'blue')
plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.legend(['Observed', 'Aladin v∞'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('galaxy_rotation_aladin.png', dpi=300)
plt.close()
