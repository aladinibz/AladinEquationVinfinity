
# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt

# --- QUATERNION SPIN (i,j,k) ---
v = np.linspace(50, 300, 100)
M = v**3 / (4.3e-3 * 1.2e-10)  # v³ scaling

print(f"Spin Mass = {M[-1]:.1e} M⊙ @ 300 km/s")

# --- PLOT ---
plt.figure(figsize=(6,4))
plt.plot(v, M / 1e10, 'cyan', lw=3)
plt.xlabel('v (km/s)')
plt.ylabel('Mass (10¹⁰ M⊙)')
plt.title('Aladin v∞ — Galaxy Spin from Quaternion Rotation')
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the plot (using a local filename for the script)
png_filename_in_script = 'quaternion_spin.png'
plt.savefig(png_filename_in_script, dpi=300)
print(f"PNG saved: {png_filename_in_script}")
