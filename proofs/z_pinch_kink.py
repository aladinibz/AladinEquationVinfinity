import numpy as np
import matplotlib.pyplot as plt

# Grid
r = np.linspace(0.1, 3, 400)
theta = np.linspace(0, 2*np.pi, 200)
R, TH = np.meshgrid(r, theta)

# Kink mode (m=1) — helical twist
k = 2*np.pi / 5  # wavelength
pert = 0.2 * np.cos(TH - k * R)  # helical perturbation
density = 1 + pert * np.exp(-R**2)

# Convert to Cartesian for plot
X = R * np.cos(TH)
Y = R * np.sin(TH)

plt.figure(figsize=(10,8))
plt.contourf(X, Y, density, levels=50, cmap='plasma')
plt.colorbar(label='Density')
plt.title('Z-Pinch Kink Instability (m=1 mode)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('/content/z_pinch_kink.png', dpi=300)
plt.show()
