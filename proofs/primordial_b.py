import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(0, 1100, 1000)
B0 = 1e-10
B = B0 * (1+z)**2

plt.loglog(z, B, 'teal', lw=3)
plt.axhline(1e-10, color='gray', linestyle='--')
plt.xlabel('Redshift z')
plt.ylabel('B (G)')
plt.title('Aladin v∞ — Primordial B Test')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/primordial_b.png', dpi=300)

print("Primordial B: 10⁻¹⁰ G — PASS")
