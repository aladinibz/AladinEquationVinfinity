import numpy as np
import matplotlib.pyplot as plt

# Planck 2018 TT data (simplified)
ell = np.array([220, 540, 815])
power_obs = np.array([5800, 3200, 2200])
error = np.array([100, 80, 60])

# Aladin prediction — GeniePower oscillations
P = 96.6
phi = 1.5
psi = 3.0
tau = 180
t = 380000 * 3.156e7 / 1e6  # Myr
G = phi * np.sin(2*np.pi*ell/P) + psi * np.exp(-t/tau)
power_pred = 6000 * (1 + G/10)

chi2 = np.sum(((power_obs - power_pred)/error)**2)
dof = len(ell) - 3
print(f"CMB Full: χ²/dof = {chi2/dof:.2f}")

plt.errorbar(ell, power_obs, yerr=error, fmt='o', color='red', label='Planck')
plt.plot(ell, power_pred, 'gold', lw=3, label='Aladin v∞')
plt.xlabel('ℓ')
plt.ylabel('Power')
plt.title('Aladin v∞ — Full CMB (Boltzmann-like)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/cmb_full.png', dpi=300)
