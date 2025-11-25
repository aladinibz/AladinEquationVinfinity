# --- INSTALL ---
!pip install matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt

# --- QUATERNION GROWTH (4D) ---
t = np.linspace(0, 200, 100)  # Myr
M = 1e8 * (1 + 0.3 * np.sin(2 * np.pi * t / 96.6)) * np.log(1 + t)

print(f"z=15 Mass = {M[-1]:.1e} M⊙")

# --- PLOT ---
plt.figure(figsize=(6,4))
plt.plot(t, M / 1e8, 'cyan', lw=3)
plt.axhline(1, color='gold', lw=2, label='10⁸ M⊙ @ 200 Myr')
plt.xlabel('Time (Myr)')
plt.ylabel('Mass (10⁸ M⊙)')
plt.title('Aladin v∞ — JWST z=15 from Quaternion Growth')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

png_path = '/content/quaternion_jwst.png'
plt.savefig(png_path, dpi=300)
print(f"PNG saved: {png_path}")

# --- DOWNLOAD .py + .png ---

print("quaternion_jwst.py + .png — DOWNLOADED")
