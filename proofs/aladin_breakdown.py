import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 300, 1000)
theta, phi, psi, P, tau = 2.0, 1.5, 3.0, 96.6, 180
genie = theta * np.log(1+t) + phi * np.sin(2*np.pi*t/P) + psi * np.exp(-t/tau)

plt.plot(t, genie, 'purple', lw=3)
plt.axhline(0, color='gray', lw=1)
plt.xlabel('Time (Myr)')
plt.ylabel('Genie Power')
plt.title('Aladin v∞ — Breakdown')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/aladin_breakdown.png', dpi=300)
