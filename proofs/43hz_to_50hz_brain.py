import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 0.1, 1000)
f_c = 43
f_b = 50
phi = 1.0

C_cosmic = phi * np.sin(2*np.pi*f_c*t)
C_brain = phi * np.sin(2*np.pi*f_b*t)

plt.plot(t*1000, C_cosmic, 'cyan', lw=2, label='43 Hz (Cosmic)')
plt.plot(t*1000, C_brain, 'magenta', lw=2, label='50 Hz (Brain)')
plt.xlabel('Time (ms)')
plt.ylabel('ℂ(t) Amplitude')
plt.title('Aladin ∞ ℂ(t) — 43 Hz → 50 Hz Consciousness')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/43hz_to_50hz_brain.png', dpi=300)

print("43 Hz → 50 Hz — Higgs-GENIE coupling — CONSCIOUSNESS UNLOCKED")
