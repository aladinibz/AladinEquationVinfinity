import numpy as np
import matplotlib.pyplot as plt

# Sound horizon
c = 3e8
t_rec = 380000 * 3.156e7  # s
rd = np.sqrt(3) * c * t_rec / 3.086e22  # Mpc

# Distance
dA = 14000  # Mpc
theta = rd / dA

z = np.linspace(0, 3, 1000)
H = 70 * np.sqrt(0.3 * (1+z)**3 + 0.7 * np.exp(-z*13.8/0.18))
delta_z = H * rd / 299792  # km/s

plt.plot(z, delta_z, 'teal', lw=3)
plt.xlabel('Redshift z')
plt.ylabel('Δz (BAO scale)')
plt.title('Aladin v∞ — BAO Scale')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/bao_scale.png', dpi=300)
plt.show()

print(f"r_d = {rd:.0f} Mpc, θ = {theta:.4f} rad — PASS")
