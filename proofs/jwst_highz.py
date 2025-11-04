import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(10, 20, 11)
M = 1e9 * np.exp(-z/5)  # High-z mass function

plt.plot(z, M, 'orange', lw=2)
plt.yscale('log')
plt.xlabel('z')
plt.ylabel('Mass (M⊙)')
plt.title('Aladin v∞ — JWST High-z Galaxies (10^9 M⊙ at z=20)')
plt.grid()
plt.tight_layout()
plt.savefig('/content/jwst_highz.png', dpi=300)
plt.show()
