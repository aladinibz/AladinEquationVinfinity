import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

A_s = np.linspace(0, 20, 100)  # sausage amplitude [km]
f0 = 43.0
coupling_strength = 0.005  # arbitrary scaling (0.005 Hz/km²)

delta_f = coupling_strength * A_s**2

plt.figure(figsize=(10,6),facecolor='black')
plt.plot(A_s, delta_f, color='gold', lw=3)
plt.xlabel('Sausage Amplitude A_s (km)', color='white')
plt.ylabel('Δf (Hz)', color='white')
plt.title('Nonlinear Sausage–Kink Coupling Shift', color='gold')
plt.grid(alpha=0.3)
plt.gca().set_facecolor('black')
plt.savefig('plots/nonlinear_sausage_kink_coupling.png', dpi=300, facecolor='black')
plt.close()
