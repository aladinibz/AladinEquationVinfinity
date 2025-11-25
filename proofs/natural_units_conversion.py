import numpy as np
import matplotlib.pyplot as plt

m_psi = 20.1e9  # eV
hbar = 6.582e-16
f = m_psi / hbar / (2 * np.pi)

plt.bar(['m_ψ (20.1 GeV)', 'f (43 Hz)'], [m_psi, f], color=['gold', 'cyan'])
plt.yscale('log')
plt.ylabel('Value')
plt.title('Aladin v∞ — Natural Units Conversion')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/natural_units_conversion.png', dpi=300)
