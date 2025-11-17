
# --- INSTALL ---
!pip install matplotlib -q

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt

# --- QFT BASICS — GENIE SCALAR FIELD ---
hbar = 1.0545718e-34  # J·s
c = 3e8  # m/s
f = 43  # Hz
omega = 2 * np.pi * f  # rad/s
m_genie = 20.4 * 1.78266192e-27  # GeV to kg
E = hbar * omega
p = np.sqrt(E**2 - (m_genie * c**2)**2)  # momentum

print(f"43 Hz QFT: p = {p:.2e} kg·m/s — GENIE resonance")

# --- PLOT ---
plt.figure(figsize=(8,5))
plt.plot([0, f], [0, omega], 'cyan', lw=3, marker='o', label='43 Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('ω (rad/s)')
plt.title('Aladin v∞ — 43 Hz QFT Derivation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('43hz_qft.png', dpi=300, bbox_inches='tight')
print("43hz_qft.png saved in plots/")

