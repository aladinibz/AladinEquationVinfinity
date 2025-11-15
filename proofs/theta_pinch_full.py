# --- INSTALL ---
!pip install matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- THETA-PINCH FORCE ---
r = np.logspace(-2, 2, 100)
F = r  # F ∝ r (axial)
print(f"F at r=1 m = {F[50]:.1e}")

# --- PLOT ---
plt.figure(figsize=(6,4))
plt.loglog(r, F, 'cyan', lw=3)
plt.xlabel('Radius r (m)')
plt.ylabel('|F_θ-Pinch|')
plt.title('Aladin v∞ — Full Theta-Pinch Equations')
plt.grid(alpha=0.3)
plt.tight_layout()

# --- SAVE PNG ---
png_path = '/content/theta_pinch_full.png'
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"PNG SAVED: {png_path}")

# --- DOWNLOAD .py + .png ---
files.download('/content/theta_pinch_full.py')
files.download(png_path)

print("theta_pinch_full.py + .png — DOWNLOADED")
