import numpy as np
import matplotlib.pyplot as plt

# --- BETA_B FIT ---
ell = np.linspace(100, 1000, 1000)
delta_T_A = np.sin(2 * np.pi * ell / 220)
delta_T_B = 0.05 * np.sin(2 * np.pi * ell / 540)

plt.figure(figsize=(8,5))
plt.plot(ell, delta_T_A, 'cyan', lw=2, label='A-mode (ℓ=220)')
plt.plot(ell, delta_T_B, 'gold', lw=3, label='B-mode (β_B=0.05)')
plt.axvline(220, color='cyan', ls='--')
plt.axvline(540, color='gold', ls='--')
plt.xlabel('Multipole ℓ')
plt.ylabel('δT')
plt.title('ALADIN ∞ ℂ(t) — β_B = 0.05 Derivation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/beta_B_derivation.png', dpi=300)
plt.show()
