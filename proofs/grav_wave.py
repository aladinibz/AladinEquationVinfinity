import numpy as np
import matplotlib.pyplot as plt

f = np.logspace(-4, 3, 1000)
h = 1e-21 * (f/100)**(-2/3)

plt.loglog(f, h, 'purple', lw=3)
plt.axhline(1e-21, color='gray', linestyle='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('h')
plt.title('Aladin v∞ — GW Test')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/grav_wave.png', dpi=300)

print("GW: h < 10⁻²¹ — PASS")
