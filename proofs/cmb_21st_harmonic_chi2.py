# cmb_21st_harmonic_chi2.py
# ALADIN ∞ ℂ(t) — chi² for 21st CMB harmonic
import numpy as np
import matplotlib.pyplot as plt

f_base = 43.0
n = 21
z = 1090.0
coupling = 250.0

f_emit = n * f_base
f_obs = f_emit / (1 + z) * coupling

l_21 = 5670
f_obs_planck = l_21 / (np.pi * 0.01) / (1 + z)

residual = (f_obs - f_obs_planck) / f_obs_planck
chi2_21 = residual**2

print(f"21st effective: {f_obs:.1f} Hz")
print(f"Planck observed: {f_obs_planck:.1f} Hz")
print(f"chi² (21st): {chi2_21:.4f}")

# Plot residuals (n=1 to 21)
n_arr = np.arange(1, 22)
f_eff_arr = n_arr * f_base / (1 + z) * coupling
f_planck_arr = 220 * n_arr / (np.pi * 0.01) / (1 + z)
res_arr = (f_eff_arr - f_planck_arr) / f_planck_arr

plt.plot(n_arr, res_arr, 'o-', color='#1f77b4')
plt.axvline(21, color='red', ls='--')
plt.xlabel('Harmonic n')
plt.ylabel('Residual')
plt.title('CMB Harmonics Residuals (21st marked)')
plt.grid(True)
plt.savefig('cmb_21st_harmonic_chi2.png', dpi=300, bbox_inches='tight')
plt.close()
