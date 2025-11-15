import numpy as np
import matplotlib.pyplot as plt

v = 246.0
y_c = 0.163
m_psi = y_c * v / np.sqrt(2)

plt.bar(['Higgs VEV', 'GENIE m_ψ'], [v, m_psi], color=['red', 'gold'])
plt.ylabel('Energy (GeV)')
plt.title('Aladin v∞ — m_ψ = 20.1 GeV from Z-Pinch')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/m_psi_zpinch.png', dpi=300)
plt.show()
