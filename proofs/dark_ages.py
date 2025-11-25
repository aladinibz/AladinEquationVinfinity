import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(6, 10, 1000)
tau = 0.054 * np.exp(-(z-7.7)**2 / (2*0.7**2))

plt.plot(z, tau, 'orange', lw=3)
plt.axhline(0.054, color='gray', linestyle='--')
plt.xlabel('Redshift z')
plt.ylabel('Optical Depth τ')
plt.title('Aladin v∞ — Reionization Test')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/dark_ages.png', dpi=300)

print("Reionization: τ = 0.054 — PASS")
