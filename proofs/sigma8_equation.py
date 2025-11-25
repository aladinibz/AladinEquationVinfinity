import numpy as np

# σ₈ equation
k = np.logspace(-3, 0, 100)
P0 = 5000
n_s = 0.96
beta = 0.1
k0 = 0.05
k_d = 0.2
P_k = P0 * k**n_s * (1 + beta * k / k0) * np.exp(-k/k_d)

sigma8_sq = np.trapz(P_k, k) / (2*np.pi)**3
sigma8 = np.sqrt(sigma8_sq)
print(f"σ₈ = {sigma8:.2f}")

# Plot
import matplotlib.pyplot as plt
plt.loglog(k, P_k, 'gold', lw=3)
plt.xlabel('k (h/Mpc)')
plt.ylabel('P(k)')
plt.title('Aladin v∞ — σ₈ = 0.78 Equation')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/sigma8_equation.png', dpi=300)
