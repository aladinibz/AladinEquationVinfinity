
# --- INSTALL ---
!pip install matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- SIMULATED PLANCK DATA (for demo) ---
ell = np.arange(30, 2500, 50)
power_obs = 5000 * np.sin(ell / 220) * np.exp(-ell / 1000) + 2000
error = power_obs * 0.1

# --- ALADIN PREDICTION ---
P = 96.6
phi = 1.5
theta_val = 2.0
psi = 3.0
tau = 180
t_rec = 380000 * 3.156e7 / 1e6  # Myr
G = theta_val * np.log(1 + ell / 1000) + phi * np.sin(2 * np.pi * ell / P) + psi * np.exp(-t_rec / tau)
power_aladin = 5000 * (1 + G / 10) * np.exp(-ell / 1000)

# --- χ² ---
chi2 = np.sum(((power_obs - power_aladin) / error)**2)
dof = len(ell) - 3
print(f"CMB Multipoles χ²/dof = {chi2/dof:.2f}")

# --- PLOT ---
plt.figure(figsize=(10, 6))
plt.errorbar(ell, power_obs, yerr=error, fmt='o', color='red', capsize=3, label='Planck (simulated)')
plt.plot(ell, power_aladin, 'gold', lw=3, label='Aladin v∞')
plt.xlabel('ℓ')
plt.ylabel('C_ℓ (μK²)')
plt.title('Aladin v∞ — CMB Multipoles')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# --- SAVE + DOWNLOAD ---
plt.savefig('/content/cmb_multipoles.png', dpi=300, bbox_inches='tight')
plt.show()
files.download('/content/cmb_multipoles.png')
files.download('/content/cmb_multipoles.py')  # Save script too

print("cmb_multipoles.py + .png saved + downloaded")
