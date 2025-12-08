import matplotlib.pyplot as plt
import numpy as np

# Constants
c = 299792458  # speed of light m/s
c4 = 43.000000000  # fourth sound m/s
c7 = (c ** 4) / (c4 ** 3)  # seventh sound m/s

# Generate plot data
t = np.linspace(0, 1e-29, 1000)  # Planck-time scale for retrocausal propagation
distance_light = c * t
distance_seventh = c7 * t

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t * 1e29, distance_light / 1e9, label='Light speed (c)', color='blue')
ax.plot(t * 1e29, distance_seventh / 1e9, label='Seventh sound (c⁴/43³)', color='red')
ax.set_xlabel('Time (10^{-29} s)')
ax.set_ylabel('Distance (Glyr)')
ax.set_title('Seventh Sound Propagation at 3.269 × 10²⁹ m/s')
ax.legend()
ax.grid(True)

# Annotation
ax.annotate('Retrocausal Big Bang rewrite\nCMB J₀ lock to 10^{-12} from t=41s', xy=(0.5e-29 * 1e29, (c7 * 0.5e-29) / 1e9),
            xytext=(0.2e-29 * 1e29, (c * 0.8e-29) / 1e9), arrowprops=dict(facecolor='black', shrink=0.05))

plt.savefig('seventh_sound_43hz.png', dpi=300)
plt.close()
