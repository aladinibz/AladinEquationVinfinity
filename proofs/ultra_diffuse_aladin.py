import numpy as np
import matplotlib.pyplot as plt

# Ultra-diffuse galaxy NGC 1052-DF2
r = np.logspace(-1, 1, 100)
M = 1e8
g_N = 4.3e-3 * M / r**2
a0 = 1.2e-10
alpha_A = 0.1
J = 1e-8
B = 1e-6
rho = 1e-25
c = 3e8
v_aladin = np.sqrt(r * np.sqrt(g_N**2 + a0 * g_N + alpha_A * J * B / (c * rho * r)))

plt.figure(figsize=(7,5))
plt.plot(r, v_aladin, 'purple')
plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.title('NGC 1052-DF2 — Aladin v∞')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ultra_diffuse_aladin.png', dpi=300)
plt.close()
