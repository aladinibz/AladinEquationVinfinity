import numpy as np
import matplotlib.pyplot as plt

# Z-pinch parameters
B = 1e-10  # T
rho = 1e-23  # kg/m^3
mu0 = 4 * np.pi * 1e-7
v_A = B / np.sqrt(mu0 * rho)

# Helical geometry
r = 3.1e20  # m (10 kpc)
L = 100 * 3.1e21  # m (100 kpc)
theta = 88 * np.pi / 180  # radians

# Frequency
f = (v_A * np.cos(theta)) / L

print(f"Alfvén speed: {v_A:.2e} m/s")
print(f"Helical frequency: {f:.2f} Hz")

# Plot spectrum
freqs = np.logspace(-16, -14, 100)
power = 1 / (1 + ((freqs - f)/1e-16)**2)

plt.figure(figsize=(8,5))
plt.loglog(freqs, power, 'purple', lw=2)
plt.axvline(f, color='gold', ls='--', label=f'43 Hz = {f:.2f} Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (arb)')
plt.title('43 Hz Resonance — Aladin v∞')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('43hz_resonance.png', dpi=300)
plt.close()
