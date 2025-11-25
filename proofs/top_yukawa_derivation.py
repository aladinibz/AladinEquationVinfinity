import numpy as np
import matplotlib.pyplot as plt

# --- TOP YUKAWA DERIVATION ---
v = 246
m_t = 172.69
y_t = np.sqrt(2) * m_t / v

plt.figure(figsize=(8,5))
plt.bar(['Top Quark'], [y_t], color='gold')
plt.ylabel('Yukawa Coupling y_t')
plt.title('ALADIN ∞ ℂ(t) — Top Yukawa Derivation')
plt.text(0, y_t/2, f'{y_t:.3f}', ha='center', va='center', color='black', fontweight='bold')
plt.ylim(0, 1.1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/top_yukawa_derivation.png', dpi=300)
