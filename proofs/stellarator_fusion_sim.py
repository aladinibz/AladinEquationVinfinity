
import numpy as np
import matplotlib.pyplot as plt

# --- HELICAL GRID (R, φ, Z) ---
R = np.linspace(1.5, 3.5, 100)
phi = np.linspace(0, 2*np.pi, 100)
R, phi = np.meshgrid(R, phi)
X = R * np.cos(phi)
Y = R * np.sin(phi)
Z = 0.5 * np.sin(5 * phi)  # 5-fold twist

# --- STELLARATOR FIELD ---
B = 1 + 0.1 * np.sin(5 * phi)
F = B * np.gradient(B, axis=0)

# --- PLOT ---
plt.figure(figsize=(8,6))
plt.contourf(X, Y, F, cmap='plasma')
plt.colorbar(label='∇B Force')
plt.title('Aladin v∞ — Stellarator Fusion: 5-Fold Twist')
plt.xlabel('X'); plt.ylabel('Y')
plt.grid(alpha=0.3)
plt.tight_layout()

png_path = 'stellarator_fusion_sim.png'
plt.savefig(png_path, dpi=300)
plt.show()
print(f"PNG saved: {png_path}")
