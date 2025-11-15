import numpy as np
import matplotlib.pyplot as plt

# --- OCTONION BASIS (7 IMAGINARIES) ---
e = np.arange(1, 8)
f = 43 * (1 + 0.1 * np.sin(2 * np.pi * e / 7))

print(f"43 Hz Octonion Braid: {f.mean():.1f} Hz")

# --- PLOT ---
plt.figure(figsize=(6,4))
plt.plot(e, f, 'gold', marker='o', lw=3)
plt.axhline(43, color='cyan', lw=2, label='43 Hz')
plt.xlabel('Octonion Basis (e1–e7)')
plt.ylabel('f (Hz)')
plt.title('Aladin v∞ — 43 Hz from Octonion Braid')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Note: plt.savefig() and plt.show() are often handled separately
# when running a script, but included here for completeness if desired.
# plt.savefig('octonion_43hz.png', dpi=300)
plt.show()