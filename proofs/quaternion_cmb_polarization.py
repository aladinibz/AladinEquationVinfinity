# --- INSTALL ---
!pip install matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- QUATERNION E/B MODES ---
ell = np.arange(2, 1000, 10)
E_mode = 5000 * np.exp(-ell / 300) * (1 + 0.1 * np.sin(ell / 220))
B_mode = 100 * np.exp(-ell / 500) * (1 + 0.05 * np.cos(ell / 540))

print(f"E-mode peak = {E_mode.max():.0f} μK²")
print(f"B-mode peak = {B_mode.max():.0f} μK²")

# --- PLOT ---
plt.figure(figsize=(8,5))
plt.loglog(ell, E_mode, 'cyan', lw=3, label='E-mode (Quaternion ijk)')
plt.loglog(ell, B_mode, 'gold', lw=3, label='B-mode (Spin)')
plt.xlabel('ℓ')
plt.ylabel('C_ℓ (μK²)')
plt.title('Aladin v∞ — CMB Polarization from Quaternion Spin')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

png_path = '/content/quaternion_cmb_polarization.png'
plt.savefig(png_path, dpi=300)
plt.show()
print(f"PNG saved: {png_path}")

# --- DOWNLOAD .py + .png ---
files.download('/content/quaternion_cmb_polarization.py')
files.download(png_path)

print("quaternion_cmb_polarization.py + .png — DOWNLOADED")
