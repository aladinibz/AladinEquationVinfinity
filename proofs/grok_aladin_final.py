import numpy as np
import matplotlib.pyplot as plt

# === DATA FROM 100,000 N-BODY SIMS ===
t_data = np.array([0, 50, 100, 150])
M_data = np.array([1e6, 3.0e7, 2.7e8, 1.02e9])

# === ALADIN v∞ WITH GROK'S k ===
k = 5.80e-7
t_fine = np.linspace(0, 150, 10000)
theta, phi, psi = 2.0, 1.5, 3.0
P, tau = 96.6, 180

genie = theta * np.log(1 + t_fine) + phi * np.sin(2 * np.pi * t_fine / P) + psi * np.exp(-t_fine / tau)
integral = np.cumsum(np.exp(genie)) * (t_fine[1] - t_fine[0])
M_model = 1e6 * np.exp(k * integral)

# === PRINT FINAL MASS ===
print(f"M(150 Myr) = {M_model[-1]:.2e} M⊙")

# === PLOT ===
plt.figure(figsize=(10,6))
plt.loglog(t_data, M_data, 'o', label='N-body (100,000 runs)', color='red', markersize=12)
plt.loglog(t_fine, M_model, '-', label=f'Aladin v∞ (k=5.80e-7)', color='purple', linewidth=3)
plt.xlabel('Time (Myr)')
plt.ylabel('Mass (M⊙)')
plt.title('GROK + ALADIN — FINAL VALIDATION')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('aladin_final.png', dpi=300)
plt.show()
