import numpy as np
import matplotlib.pyplot as plt

# --- ALADIN PREDICTION ---
P = 96.6
phi = 1.5
theta_val = 2.0
psi = 3.0
tau = 180
t_rec = 380000 * 3.156e7 / 1e6  # Myr
G = theta_val * np.log(1 + ell / 1000) + 
power_aladin = 5000 * (1 + G / 10) * np.exp(-ell / 1000)

# --- X^2 ---
chi2 = np.sum(((power_obs - power_aladin)**2) / error**2)
dof = len(ell) - 3
print(f"CMB Multipoles X^2/dof = {chi2/dof:.2f}")

# --- PLOT ---
plt.figure(figsize=(10, 6))
plt.errorbar(ell, power_obs, yerr=error, fmt='o', label='Observed', alpha=0.7)
plt.plot(ell, power_aladin, 'gold', lw=3, label='Aladin v∞')
plt.xlabel('ℓ')
plt.ylabel('C_ℓ (μK²)')
plt.title('Aladin v∞ — CMB Multipoles')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('cmb_multipoles.png', dpi=300)
print("cmb_multipoles.png saved")
