import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0.1, 5, 500)
I = 1e18
mu0 = 4*np.pi*1e-7
n0 = 1e6
B = mu0 * I / (2 * np.pi * r)
F = -n0 * B**2 / (mu0 * r)

plt.figure(figsize=(10,6))
plt.loglog(r, B, 'blue', label='B-field')
plt.loglog(r, -F, 'red', label='Pinch Force')
plt.xlabel('Radius (cm)')
plt.ylabel('Strength')
plt.title('Z-Pinch Mechanism — B-field + Pinch Force')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/z_pinch_mechanism.png', dpi=300)
