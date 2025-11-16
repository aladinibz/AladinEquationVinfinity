import numpy as np
import matplotlib.pyplot as plt

# --- m_ψ FROM PLASMA DOUBLET ---
y_c = 0.163
v = 246
m_psi = y_c * v / 2

plt.figure(figsize=(8,5))
plt.bar(['Plasma Doublet'], [m_psi], color='gold')
plt.ylabel('m_ψ (GeV)')
plt.title('ALADIN ∞ ℂ(t) — m_ψ = 20.05 GeV from Plasma')
plt.text(0, m_psi/2, f'{m_psi:.2f} GeV', ha='center', va='center', color='black', fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/m_psi_plasma.png', dpi=300)
plt.show()
