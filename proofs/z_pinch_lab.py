import numpy as np
import matplotlib.pyplot as plt

# --- LAB VS COSMIC ---
r = np.logspace(-3, 25, 100)
F_lab = 0.1 * (1e24 * 1e4) / (3e8 * 1e-5 * r[:50])
F_cosmic = 0.1 * (1e18 * 1e-6) / (3e8 * 1e-24 * r[50:])

plt.figure(figsize=(8,5))
plt.loglog(r[:50], F_lab, 'cyan', lw=3, label='Lab (Sandia)')
plt.loglog(r[50:], F_cosmic, 'gold', lw=3, label='Cosmic (Galaxy)')
plt.xlabel('Radius r (m)')
plt.ylabel('F_Z-Pinch (N/m³)')
plt.title('ALADIN ∞ ℂ(t) — Z-Pinch Lab vs Cosmic')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/z_pinch_lab.png', dpi=300)
