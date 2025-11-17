import numpy as np
import matplotlib.pyplot as plt

# --- THETA PINCH STABILITY ---
k = np.logspace(-3, 1, 100)
B_theta = 0.8e-6
B_z = 1e-6
cs = 1e5
omega2 = k**2 * cs**2 * (1 - (B_theta/B_z)**2)

plt.figure(figsize=(8,5))
plt.loglog(k, omega2, 'gold', lw=3)
plt.axhline(0, color='red', ls='--')
plt.xlabel('Wavenumber k')
plt.ylabel('ω²')
plt.title('ALADIN ∞ C(t) — Theta Pinch Stability')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('theta_pinch_stability.png', dpi=300)
print("theta_pinch_stability.png saved")
