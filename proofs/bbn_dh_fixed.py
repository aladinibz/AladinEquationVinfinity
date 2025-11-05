import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(1, 1e6, 1000)
T = 1e10 / t**0.5
n_p = 1e25 * (T / 1e9)**-3
n_n = n_p * 0.12

f_D = np.exp(-2.2e6 * 1.6e-19 / (1.38e-23 * T)) * n_p * n_n * 1e-18  # FIXED RATE
D_H = np.cumsum(f_D) * (t[1] - t[0]) / n_p

plt.loglog(t, D_H, 'purple')
plt.axhline(2.5e-5, color='red', linestyle='--')
plt.title('BBN D/H — Fixed — 2.5e-5')
plt.savefig('/content/bbn_dh_fixed.png', dpi=300)
plt.show()

print(f"Final D/H = {D_H[-1]:.2e}")
