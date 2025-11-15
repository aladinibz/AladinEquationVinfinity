
import numpy as np
import matplotlib.pyplot as plt

# --- GRID ---
nx, ny = 100, 100
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2 + 1e-6)

# --- Z-PINCH + THETA-PINCH ---
J_z = 1e18 * np.exp(-r**2 / 0.1)
J_theta = 1e17 * np.exp(-r**2 / 0.2)
F_r = -0.1 * 2 * np.pi * J_z**2 * r
F_z = 0.05 * 2 * np.pi * J_theta**2 * r

# --- PLOT ---
plt.figure(figsize=(8,6))
plt.streamplot(X, Y, F_r * X/r, F_r * Y/r, color='gold', linewidth=1, density=2)
plt.title('Aladin v∞ — MHD Plasma: Z-Pinch + Theta-Pinch')
plt.xlabel('x'); plt.ylabel('y')
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the plot
png_path_for_script = 'mhd_plasma_sim.png'
plt.savefig(png_path_for_script, dpi=300)
plt.show()
print(f"PNG saved: {png_path_for_script}")
