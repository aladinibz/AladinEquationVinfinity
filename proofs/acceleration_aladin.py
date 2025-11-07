import numpy as np
import matplotlib.pyplot as plt

z = np.logspace(-1, 0.5, 100) - 0.9
dL_planck = 1e4 * z * (1 + z) / (1 + z**2)
dL_aladin = 1e4 * np.log(1 + z) * np.sqrt(1 + z)

plt.figure(figsize=(7,5))
plt.plot(z, dL_planck, 'k--')
plt.plot(z, dL_aladin, 'green')
plt.xscale('log')
plt.xlabel('Redshift z')
plt.ylabel('Luminosity Distance (Mpc)')
plt.legend(['Planck ΛCDM', 'Aladin v∞'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('acceleration_aladin.png', dpi=300)
plt.close()
