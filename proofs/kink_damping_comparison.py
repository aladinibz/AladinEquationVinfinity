import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Parameters
f0 = 43.0
omega0 = 2 * np.pi * f0
t = np.logspace(-1, 3, 200)  # 0.1 s to 1000 s

# Resonant absorption (exponential)
rho_ratio = 10.0; delta_L = 0.1
contrast = (rho_ratio-1)/(rho_ratio+1)
gamma_ra = -(np.pi/2) * contrast * delta_L * omega0
A_ra = np.exp(gamma_ra * t)

# Phase mixing (power-law)
delta_omega_ratio = 0.4
gamma_pm = -(1/3) * (delta_omega_ratio)**(2/3) / t
A_pm = (t / t[0])**(-1/3)

# Normalize to start at 1
A_ra /= A_ra[0]
A_pm /= A_pm[0]

plt.figure(figsize=(12,7), facecolor='black')
plt.plot(t, A_ra, color='gold', lw=3, label='Resonant Absorption (exp)')
plt.plot(t, A_pm, color='lime', lw=3, label='Phase Mixing (t^{-1/3})')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Time (s)', color='white')
plt.ylabel('Normalized Amplitude', color='white')
plt.title('Kink Damping: Resonant vs Phase Mixing', color='gold')
plt.legend(loc='upper right', framealpha=0.2)
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/kink_damping_comparison.png', dpi=300, facecolor='black')
plt.close()
