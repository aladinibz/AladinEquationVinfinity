import numpy as np
import matplotlib.pyplot as plt

# Hubble tension
z = np.logspace(-1, 1, 100) - 0.9
H0 = 73
H_aladin = H0 * np.log(1 + z)

plt.figure(figsize=(7,5))
plt.plot(z, H_aladin, 'gold')
plt.xlabel('Redshift z')
plt.ylabel('H(z) (km/s/Mpc)')
plt.title('Hubble Tension — Aladin v∞')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hubble_tension_aladin.png', dpi=300)
plt.close()
