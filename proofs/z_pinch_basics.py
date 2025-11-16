import numpy as np
import matplotlib.pyplot as plt

r = np.logspace(18, 25, 100)
F = 0.1 * (1e18 * 1e-6) / (3e8 * 1e-24 * r)

plt.loglog(r, F, 'gold', lw=3)
plt.axvline(1e21, color='cyan', ls='--', label='Galaxy')
plt.xlabel('Radius r (m)')
plt.ylabel('F_Z-Pinch (N/m³)')
plt.title('ALADIN ∞ ℂ(t) — Z-Pinch Cosmology Basics')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/z_pinch_basics.png', dpi=300)
plt.show()
