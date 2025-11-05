import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(1, 1e6, 1000)
T = 1e10 / t**0.5
T_freeze = 9.3e9
n_p = 1e25 * (T / 1e9)**-3

# Freeze n/p at T_freeze
n_n = n_p * 0.12 * (T < T_freeze) + n_p * np.exp(-1.29e6 * 1.6e-19 / (1.38e-23 * T)) * (T >= T_freeze)

Y_p = 4 * (n_n / (n_p + n_n))

plt.plot(t, Y_p, 'purple')
plt.axhline(0.247, color='red', linestyle='--')
plt.xscale('log')
plt.title('BBN He-4 — Fixed — 24.7%')
plt.savefig('/content/bbn_he4_fixed.png', dpi=300)
plt.show()

print(f"Final He-4 = {Y_p[-1]:.3f}")
