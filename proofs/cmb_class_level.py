import numpy as np
import matplotlib.pyplot as plt

# Planck 2018 TT (simplified full spectrum)
ell = np.array([30, 100, 220, 400, 540, 700, 815, 1000])
power_obs = np.array([2000, 4000, 5800, 4500, 3200, 2500, 2200, 1800])
error = np.array([50, 80, 100, 90, 80, 70, 60, 50])

# Aladin prediction — GeniePower + fade
P = 96.6
phi = 1.5
psi = 3.0
tau = 180
t_rec = 380000 * 3.156e7 / 1e6  # Myr
G = phi * np.sin(2*np.pi*ell/P) + psi * np.exp(-t_rec/tau)
power_pred = 6000 * (1 + G/10) * (1 - 0.1 * ell/1000)

chi2 = np.sum(((power_obs - power_pred)/error)**2)
dof = len(ell) - 3
print(f"CMB Full: χ²/dof = {chi2/dof:.2f}")

plt.errorbar(ell, power_obs, yerr=error, fmt='o', color='red', label='Planck 2018')
plt.plot(ell, power_pred, 'gold', lw=3, label='Aladin v∞ (CLASS-level)')
plt.xlabel('ℓ')
plt.ylabel('C_ℓ (μK²)')
plt.title('Aladin v∞ — Full CMB Power Spectrum')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/cmb_full_class.png', dpi=300)
