import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0.1, 100, 1000)
alpha_A = 0.1
P = 96.6
shear = alpha_A * np.sin(2 * np.pi * np.log10(r) / P) * np.exp(-r / 50)

plt.loglog(r, shear, 'purple')
plt.title('Euclid Weak Lensing — Aladin v∞')
plt.xlabel('r (Mpc)')
plt.ylabel('Shear')
plt.savefig('/content/euclid_lensing.png', dpi=300)
