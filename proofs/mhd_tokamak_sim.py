# --- INSTALL ---
!pip install matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt

# --- TOROIDAL GRID (R, Z) ---
R = np.linspace(1.5, 3.5, 100)
Z = np.linspace(-1, 1, 100)
R, Z = np.meshgrid(R, Z)
r = np.sqrt((R - 2.5)**2 + Z**2)

# --- Z-PINCH + THETA-PINCH ---
J_phi = 1e18 * np.exp(-r**2 / 0.3)
B_theta = 2 * np.pi * J_phi * r / 3e8
F_r = -0.1 * J_phi * B_theta

# --- PLOT ---
plt.figure(figsize=(8,6))
plt.contourf(R, Z, F_r, cmap='plasma')
plt.colorbar(label='F_r (N/m³)')
plt.title('Aladin v∞ — MHD Tokamak: Z-Pinch + Theta-Pinch')
plt.xlabel('R (m)'); plt.ylabel('Z (m)')
plt.grid(alpha=0.3)
plt.tight_layout()

png_path = '/content/mhd_tokamak_sim.png'
plt.savefig(png_path, dpi=300)
print(f"PNG saved: {png_path}")

# --- DOWNLOAD .py + .png ---

print("mhd_tokamak_sim.py + .png — DOWNLOADED")
