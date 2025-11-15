import numpy as np
import matplotlib.pyplot as plt

r = np.logspace(-2, 2, 100)
B = 1 / r  # Z-Pinch Bθ
J = 1e18
force = J * B

plt.loglog(r, force, 'gold', lw=3)
plt.xlabel('Radius r (m)')
plt.ylabel('J × B Force')
plt.title('Aladin v∞ — Z-Pinch Symmetry (m_0 = 0)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/z_pinch_symmetry.png', dpi=300)
plt.show()
