import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(10, 20, 100)
M = 1e6 * (1 + z)**3.5  # Aladin v∞ high-z growth

plt.plot(z, M, 'purple')
plt.yscale('log')
plt.title('JWST z=20 Outliers — Aladin v∞')
plt.xlabel('Redshift z')
plt.ylabel('Mass (M⊙)')
plt.savefig('/content/jwst_z20_outliers.png', dpi=300)
