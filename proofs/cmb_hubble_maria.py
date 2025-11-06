import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 300, 1000)
H0 = 70
Om = 0.3
OL = 0.7
tau_A = 180
H_Maria = H0 * np.sqrt(Om * (1+t)**(-3) + OL * np.exp(-t/tau_A))

plt.plot(t, H_Maria, 'gold', lw=3)
plt.xlabel('Time (Myr)')
plt.ylabel('H_Maria(t) (km/s/Mpc)')
plt.title('Aladin v∞ — H_Maria(t) Expansion')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/cmb_hubble_maria.png', dpi=300)
plt.show()
