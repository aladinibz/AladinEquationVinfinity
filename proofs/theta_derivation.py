import numpy as np
import matplotlib.pyplot as plt

# --- THETA FIT ---
t = np.logspace(1, 3, 100)  # Myr
M0 = 1e6
theta = 0.8
M = M0 * (1 + t)**theta

plt.figure(figsize=(8,5))
plt.loglog(t, M, 'gold', lw=3)
plt.scatter(150, 1e9, color='cyan', s=100, label='JWST z=20')
plt.xlabel('Time t (Myr)')
plt.ylabel('Mass M (M⊙)')
plt.title('ALADIN ∞ ℂ(t) — θ = 0.8 Derivation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/theta_derivation.png', dpi=300)
