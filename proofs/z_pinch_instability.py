import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0.1, 5, 500)
z = np.linspace(0, 10, 500)
R, Z = np.meshgrid(r, z)

# Sausage mode (m=0)
pert = 0.1 * np.cos(2 * np.pi * Z / 3)
density = 1 + pert * np.exp(-R**2)

plt.figure(figsize=(10,6))
plt.contourf(R, Z, density, cmap='plasma')
plt.colorbar(label='Density')
plt.title('Z-Pinch Sausage Instability')
plt.xlabel('Radius')
plt.ylabel('Z (along current)')
plt.savefig('/content/z_pinch_instability.png', dpi=300)
