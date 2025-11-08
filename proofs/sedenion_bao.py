# /proofs/sedenion_bao.py
import numpy as np
import matplotlib.pyplot as plt

# Comoving distance
r = np.linspace(50, 200, 1000)

# 43 Hz → k0
c_s = 0.58 * 3e5  # km/s
k0 = 2 * np.pi * 43 / c_s

# 15 sedenion harmonics
e = np.linspace(1.0, 0.1, 15)
P_k = np.zeros_like(r)
for i in range(15):
    k = k0 * (i + 1)
    P_k += e[i] * np.sin(k * r)**2

# Aladin v∞ damping
P_aladin = P_k * np.exp(-r / 150)

# Plot
plt.figure(figsize=(12,7))
plt.plot(r, P_k, 'purple', lw=2, label='Sedenion 15 BAO')
plt.plot(r, P_aladin, 'gold', lw=3, label='Aladin v∞')
plt.xlabel('r [Mpc/h]'); plt.ylabel('P(k)')
plt.title('Sedenion 15-Mode BAO')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/sedenion_bao_15mode.png', dpi=300)
plt.close()

print("BAO plot saved")
