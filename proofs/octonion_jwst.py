
import numpy as np
import matplotlib.pyplot as plt

# --- OCTONION GROWTH (7D) ---
t = np.linspace(0, 150, 100)  # Myr
M = 1e9 * (1 + 0.5 * np.sin(2 * np.pi * t / 96.6)) * np.log(1 + t)
print(f"z=20 Mass = {M[-1]:.1e} M⊙")

# --- PLOT ---
plt.figure(figsize=(6,4))
plt.plot(t, M / 1e9, 'gold', lw=3)
plt.axhline(1, color='cyan', lw=2, label='10⁹ M⊙ @ 150 Myr')
plt.xlabel('Time (Myr)')
plt.ylabel('Mass (10⁹ M⊙)')
plt.title('Aladin v∞ — JWST z=20 from Octonion Growth')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

png_path = 'octonion_jwst.png'
plt.savefig(png_path, dpi=300)
plt.show()
