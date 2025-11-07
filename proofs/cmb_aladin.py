import numpy as np
import matplotlib.pyplot as plt

l = np.arange(2, 2500)
cl_planck = 1e-7 * l * (l+1) * np.exp(-l/1000) * (1 + np.sin(l/200))
cl_aladin = 1e-7 * l * (l+1) * np.exp(-l/100) * np.sin(l/10)

plt.figure(figsize=(8,5))
plt.plot(l, cl_planck, 'gray', alpha=0.7)
plt.plot(l, cl_aladin, 'red')
plt.xlim(2, 2500)
plt.xlabel('Multipole l')
plt.ylabel('C_l')
plt.legend(['Planck 2018', 'Aladin v∞'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cmb_aladin.png', dpi=300)
plt.close()
