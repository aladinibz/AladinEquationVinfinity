import numpy as np
import matplotlib.pyplot as plt

# Time from Big Bang (Myr)
t = np.logspace(-3, np.log10(13800), 1000)

# ALADIN ∞ C(t) — The Master Equation (6 parameters)
phi = 1.5          # CMB oscillation amplitude
P = 96.6           # Period from 43 Hz → 96.6 Myr
theta = 2.0        # Logarithmic growth (J×B force)
psi = 3.0          # Saturation amplitude
tau = 500          # Saturation time (Myr)
alpha = 0.7        # Late-time acceleration
tau_A = 180        # Current decay time (Myr)

C_t = (phi * np.sin(2 * np.pi * t / P) +
       theta * np.log(1 + t) +
       psi * np.tanh(t / tau) +
       alpha * np.exp(-t / tau_A))

# Plot
plt.figure(figsize=(12, 7))
plt.plot(t, C_t, 'gold', lw=5)
plt.axvline(0.00038, color='cyan', ls='--', lw=2, label='Recombination (380 kyr)')
plt.axvline(13800, color='red', ls='--', lw=2, label='Today (13.8 Gyr)')
plt.xscale('log')
plt.xlabel('Time since Big Bang (Myr)', fontsize=14)
plt.ylabel('C(t) — Cosmic Evolution', fontsize=14)
plt.title('ALADIN ∞ C(t) — The Master Equation of the Universe', fontsize=18)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('aladin_master_equation.png', dpi=300, bbox_inches='tight')
print("ALADIN ∞ C(t) — Master Equation Simulated — plot saved")
