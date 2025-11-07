import numpy as np
import matplotlib.pyplot as plt

# Abell 1689 lensing
theta = np.linspace(0, 2*np.pi, 100)
r = 100 + 20*np.sin(5*theta)  # simplified mass profile
kappa_aladin = 0.8 * np.exp(-r/50)

plt.figure(figsize=(7,7))
plt.polar(theta, r, 'purple')
plt.fill(theta, r, alpha=0.3, color='purple')
plt.title('Abell 1689 — Aladin v∞ Lensing')
plt.tight_layout()
plt.savefig('abell1689_lensing_aladin.png', dpi=300)
plt.close()
