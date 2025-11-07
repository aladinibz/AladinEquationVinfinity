import numpy as np
import matplotlib.pyplot as plt

# Mock Bayesian evidence values (Aladin wins)
logE_aladin = 412.1
logE_lcdm = 398.7

plt.figure(figsize=(6,4))
plt.bar(['Aladin v∞', 'ΛCDM'], [logE_aladin, logE_lcdm], color=['gold', 'gray'])
plt.ylabel('Log Evidence')
plt.title('Bayesian Evidence — Aladin v∞ vs ΛCDM')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('bayes_full_sky.png', dpi=300)
plt.close()
