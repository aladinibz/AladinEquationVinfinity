import numpy as np
import matplotlib.pyplot as plt

# Tully-Fisher data
v = np.array([100, 150, 200, 250, 300])  # km/s
M_obs = np.array([8e9, 2e10, 5e10, 1e11, 2e11])  # M⊙
error = M_obs * 0.1

# Aladin prediction
a0 = 1.2e-10
G = 4.3e-3  # pc/M⊙ (km/s)^2
M_aladin = v**4 / (G * a0) * (1 + 0.1 * v/200)

chi2 = np.sum(((M_obs - M_aladin)/error)**2)
dof = len(v) - 2
print(f"Tully-Fisher χ²/dof = {chi2/dof:.2f}")

plt.errorbar(v, M_obs, yerr=error, fmt='o', color='red', label='Observed')
plt.plot(v, M_aladin, 'gold', lw=3, label='Aladin v∞')
plt.xlabel('v (km/s)')
plt.ylabel('M_* (M⊙)')
plt.title('Aladin v∞ — Tully-Fisher')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/tully_fisher.png', dpi=300)
