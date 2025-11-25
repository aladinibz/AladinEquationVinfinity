import numpy as np
import matplotlib.pyplot as plt

r = np.logspace(-2, 2, 100)
F = -r  # F ∝ -r (inward)
plt.loglog(r, np.abs(F), 'gold', lw=3)
plt.xlabel('Radius r (m)')
plt.ylabel('|F_Z-Pinch|')
plt.title('Aladin v∞ — Full Z-Pinch Equations')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/z_pinch_full.png', dpi=300)
