import numpy as np
import matplotlib.pyplot as plt

r0 = 1e-2  # m
n0 = 1e19  # m^-3
T = 1e6    # K
mu0 = 4*np.pi*1e-7
kB = 1.38e-23

I_crit = np.sqrt(2*np.pi*r0*n0*kB*T / mu0)
I = np.linspace(0.5*I_crit, 2*I_crit, 100)
omega2 = (I > I_crit) * (I**2 - I_crit**2) * 1e12  # fake scaling

plt.plot(I/1e6, omega2, 'purple', lw=3)
plt.axvline(I_crit/1e6, color='red', linestyle='--', label='Critical Current')
plt.xlabel('Current (MA)')
plt.ylabel('Growth Rate² (arb)')
plt.title('Sausage Mode Growth — I > I_crit → UNSTABLE')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/sausage_growth.png', dpi=300)
