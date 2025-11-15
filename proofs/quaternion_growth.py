import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 200, 100)
M = 1e8 * (1 + 0.3 * np.sin(2 * np.pi * t / 96.6)) * np.log(1 + t)

plt.plot(t, M / 1e8, 'cyan', lw=3)
plt.axhline(1, color='gold', lw=2, label='10⁸ M⊙ @ 200 Myr')
plt.xlabel('Time (Myr)')
plt.ylabel('Mass (10⁸ M⊙)')
plt.title('Aladin v∞ — Quaternion Growth Model')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/quaternion_growth.png', dpi=300)
plt.show()
