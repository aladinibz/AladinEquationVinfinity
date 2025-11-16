import numpy as np
import matplotlib.pyplot as plt

# --- YUKAWA COUPLING ---
y_f = np.linspace(0, 0.2, 100)
v = 246
m_f = y_f * v / np.sqrt(2)

plt.figure(figsize=(8,5))
plt.plot(y_f, m_f, 'gold', lw=3)
plt.scatter(0.163, 20.05, color='cyan', s=100, label='m_ψ = 20.05 GeV')
plt.xlabel('Yukawa Coupling y_f')
plt.ylabel('Fermion Mass m_f (GeV)')
plt.title('ALADIN ∞ ℂ(t) — Yukawa Coupling in QFT')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/yukawa_qft.png', dpi=300)
plt.show()
