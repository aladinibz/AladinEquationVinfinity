import numpy as np
import matplotlib.pyplot as plt

# Time in Myr
t = np.linspace(0, 300, 1000)

# Parameters
H0 = 70.0        # km/s/Mpc
Om = 0.3         # Matter density
OL = 0.7         # Fade density
tau_A = 180.0    # Fade timescale (Myr)

# H_Maria(t)
H_Maria = H0 * np.sqrt(Om * (1 + t)**(-3) + OL * np.exp(-t / tau_A))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, H_Maria, color='gold', linewidth=3, label='H_Maria(t)')
plt.axhline(H0, color='gray', linestyle='--', label='H0 = 70 km/s/Mpc')
plt.xlabel('Time (Myr)', fontsize=12)
plt.ylabel('H_Maria(t) (km/s/Mpc)', fontsize=12)
plt.title('Aladin v∞ — H_Maria(t) Cosmic Expansion', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig('cmb_hubble_maria.png', dpi=300, bbox_inches='tight')

print("H_Maria(t) plot saved as cmb_hubble_maria.png")
