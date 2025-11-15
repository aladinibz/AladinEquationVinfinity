# --- INSTALL ---
!pip install matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- Z-PINCH FORCE ---
r = np.logspace(-2, 2, 100)
F = 1 / r  # Simplified F ∝ 1/r
print(f"F at r=1 m = {F[50]:.1e}")

# --- PLOT ---
plt.figure(figsize=(6,4))
plt.loglog(r, F, 'gold', lw=3)
plt.xlabel('Radius r (m)')
plt.ylabel('F_Z-Pinch')
plt.title('Aladin v∞ — Z-Pinch Force Derivation')
plt.grid(alpha=0.3)
plt.tight_layout()

# --- SAVE PNG ---
png_path = '/content/z_pinch_force.png'
plt.savefig(png_path, dpi=300)
plt.show()
print(f"PNG saved: {png_path}")

# --- DOWNLOAD .py + .png ---
files.download('/content/z_pinch_force.py')
files.download(png_path)

print("z_pinch_force.py + .png — DOWNLOADED")
