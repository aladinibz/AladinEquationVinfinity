import numpy as np
import matplotlib.pyplot as plt

# --- Z-PINCH COMPRESSION ---
r = np.logspace(-3, -1, 100)
J = 1e18
B = 1e4
rho0 = 1e-5
c = 3e8
cs = 1e5
F = 0.1 * (J * B) / (c * rho0 * r)
compression = F * r / cs**2

# --- PLOT ---
plt.figure(figsize=(8,5))
plt.loglog(r, compression, 'gold', lw=3)
plt.axhline(1e5, color='cyan', ls='--', label='100,000×')
plt.xlabel('Radius r (m)')
plt.ylabel('Compression Δρ/ρ')
plt.title('ALADIN ∞ ℂ(t) — Z-Pinch Prediction: 100,000× by 2027')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/z_pinch_prediction.png', dpi=300)
