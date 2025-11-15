import numpy as np
import matplotlib.pyplot as plt

r = np.logspace(-2, 2, 100)
F = 1 / r  # Z-Pinch force
plt.loglog(r, F, 'gold', lw=3)
plt.xlabel('Radius r (m)')
plt.ylabel('F_Z-Pinch')
plt.title('Aladin v∞ — Z-Pinch Cosmology')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/z_pinch_cosmology.png', dpi=300)
plt.show()
