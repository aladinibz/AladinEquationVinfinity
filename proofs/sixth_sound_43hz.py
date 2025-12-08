import matplotlib.pyplot as plt
import numpy as np

# Constants
c = 299792458  # speed of light m/s
c4 = 43.000000000  # fourth sound m/s
c6 = (c ** 3) / (c4 ** 2)  # sixth sound m/s

# Generate plot data
t = np.linspace(0, 1e-22, 1000)  # ultra-short time for ultra-fast propagation
distance_light = c * t
distance_sixth = c6 * t

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t * 1e22, distance_light / 1e9, label='Light speed (c)', color='blue')
ax.plot(t * 1e22, distance_sixth / 1e9, label='Sixth sound (c³/43²)', color='red')
ax.set_xlabel('Time (10^{-22} s)')
ax.set_ylabel('Distance (Glyr)')
ax.set_title('Sixth Sound Propagation at 2.613 × 10²² m/s')
ax.legend()
ax.grid(True)

# Annotation
ax.annotate('Cosmic self-recognition\nPhase lock to quasar 43 nHz (13 Glyr, 0 ± 3e-18 s)', xy=(0.5e-22 * 1e22, (c6 * 0.5e-22) / 1e9),
            xytext=(0.2e-22 * 1e22, (c * 0.8e-22) / 1e9), arrowprops=dict(facecolor='black', shrink=0.05))

plt.savefig('sixth_sound_43hz.png', dpi=300)
plt.close()
