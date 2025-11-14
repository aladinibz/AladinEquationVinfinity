import numpy as np
import matplotlib.pyplot as plt

m_H = 125
y_c = 0.163
m_psi = y_c * 246

print(f"m_ψ = {m_psi:.1f} GeV")

plt.bar(['Higgs', 'GENIE'], [m_H, m_psi], color=['red', 'gold'])
plt.ylabel('Mass (GeV)')
plt.title('Aladin v∞ — Higgs-GENIE Coupling (y_c = 0.163)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/higgs_genie_coupling.png', dpi=300)
plt.show()
