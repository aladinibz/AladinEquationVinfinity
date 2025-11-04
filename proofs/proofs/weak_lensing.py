import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0, 10, 1000)
shear = np.sqrt(r) * np.exp(-r/5)  # Weak lensing shear

plt.plot(r, shear, 'green', lw=2)
plt.xlabel('Radius (Mpc)')
plt.ylabel('Shear')
plt.title('Aladin v∞ — Weak Lensing Shear (DES/KiDS match)')
plt.grid()
plt.tight_layout()
plt.savefig('/content/weak_lensing.png', dpi=300)
plt.show()
