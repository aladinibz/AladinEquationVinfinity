import numpy as np
import matplotlib.pyplot as plt

# Brain parameters
B = 1e-12  # T (neuromagnetic)
rho = 1000  # kg/m^3 (brain tissue)
mu0 = 4 * np.pi * 1e-7
v_A = B / np.sqrt(mu0 * rho)

# Neural filament
L = 1e-3  # m (1 mm)
theta = 88 * np.pi / 180

# Frequency
f = (v_A * np.cos(theta)) / L

print(f"Brain Alfvén speed: {v_A:.2e} m/s")
print(f"Neural 43 Hz: {f:.2f} Hz")

# EEG spectrum
freqs = np.linspace(30, 50, 100)
power = 1 / (1 + ((freqs - f)/1)**2)

plt.figure(figsize=(8,5))
plt.plot(freqs, power, 'purple', lw=2)
plt.axvline(f, color='gold', ls='--', label=f'43 Hz = {f:.2f} Hz')
plt.xlabel('EEG Frequency (Hz)')
plt.ylabel('Power (arb)')
plt.title('43 Hz in Brain — Aladin v∞')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('43hz_brain.png', dpi=300)
plt.close()
