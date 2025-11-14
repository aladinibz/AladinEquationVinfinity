# --- INSTALL ---
!pip install matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- PARAMETERS ---
H0 = 75.2
Om = 0.3
OL = 0.7
tau_A = 180
rd = 147  # Mpc
z = 2.5
c = 299792.458  # km/s

# --- H(z) ---
t_z = 13.8 / (1 + z)  # Gyr to Myr
H_z = H0 * np.sqrt(Om * (1 + z)**3 + OL * np.exp(-t_z / tau_A))

# --- d_A(z) ---
def d_A(z, H0, Om, OL, tau_A):
    dz = 0.01
    z_grid = np.arange(0, z + dz, dz)
    t_grid = 13.8 / (1 + z_grid)
    H_grid = H0 * np.sqrt(Om * (1 + z_grid)**3 + OL * np.exp(-t_grid / tau_A))
    integral = np.trapz(c / H_grid, z_grid)
    return integral / (1 + z)

dA = d_A(z, H0, Om, OL, tau_A)
theta_bao = rd / dA
error = 0.0002

print(f"θ_BAO(z=2.5) = {theta_bao:.4f} ± {error} rad")

# --- PLOT ---
plt.figure(figsize=(8,5))
plt.axhline(theta_bao, color='gold', lw=3, label='Aladin v∞ Prediction')
plt.fill_between([0, 3], theta_bao - error, theta_bao + error, color='gold', alpha=0.3)
plt.xlabel('Redshift z')
plt.ylabel('θ_BAO (rad)')
plt.title('Aladin v∞ — DESI Year 2 BAO Prediction (z=2.5)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# --- SAVE + DOWNLOAD ---
plt.savefig('/content/desi_y2_bao_prediction.png', dpi=300, bbox_inches='tight')
plt.show()
files.download('/content/desi_y2_bao_prediction.png')
files.download('/content/desi_y2_bao_prediction.py')  # Save script too

print("desi_y2_bao_prediction.png + .py saved + downloaded")
