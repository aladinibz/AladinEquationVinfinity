import numpy as np
import matplotlib.pyplot as plt

# Grid
r = np.linspace(0.1, 3, 400)
theta = np.linspace(0, 2*np.pi, 200)
R, TH = np.meshgrid(r, theta)

# m=2 kink mode — double helix
k = 2*np.pi / 5
pert = 0.2 * np.cos(2*TH - k * R)  # m=2
density = 1 + pert * np.exp(-R**2)

# Cartesian
X = R * np.cos(TH)
Y = R * np.sin(TH)

plt.figure(figsize=(10,8))
plt.contourf(X, Y, density, levels=50, cmap='plasma')
plt.colorbar(label='Density')
plt.title('Z-Pinch Kink Instability — m=2 Mode')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('/content/z_pinch_kink_m2.png', dpi=300)
