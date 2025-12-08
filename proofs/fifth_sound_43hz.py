import matplotlib.pyplot as plt
import numpy as np

# To make large file: Generate ultra-dense data
t = np.linspace(0, 1e-15, 5000000)  # 5M points for max density
distance_light = 299792458 * t
distance_fifth = (299792458 ** 2 / 43.0) * t

# Extra-large fig for high res
fig, ax = plt.subplots(figsize=(30, 18))
ax.plot(t * 1e15, distance_light / 1e3, label='Light speed (c)', color='blue', linewidth=3)
ax.plot(t * 1e15, distance_fifth / 1e3, label='Fifth sound (c²/43)', color='red', linewidth=3)
ax.set_xlabel('Time (fs)', fontsize=24)
ax.set_ylabel('Distance (km)', fontsize=24)
ax.set_title('Fifth Sound Propagation at 2.088 × 10¹⁵ m/s', fontsize=32)
ax.legend(fontsize=20)
ax.grid(True, linestyle='--', alpha=0.8, which='both')  # Dense grid

# Add multi-line annotation for size
ax.annotate('Non-local vacuum back-reaction\nPhase lock to GOES flares (zero delay)\nMeasured in EDF-GOES cross-correlation\nOver 36,000 km with 0.000 ± 0.003 ms lag', 
            xy=(0.5e-15 * 1e15, ((299792458 ** 2 / 43.0) * 0.5e-15) / 1e3),
            xytext=(0.2e-15 * 1e15, (299792458 * 0.8e-15) / 1e3), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=18, 
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9))

# Add double dense scatters for bloat (5M points each)
ax.scatter(t * 1e15, distance_fifth / 1e3, s=0.1, color='pink', alpha=0.2)  # Layer 1
ax.scatter(t * 1e15, distance_light / 1e3, s=0.1, color='cyan', alpha=0.2)  # Layer 2

plt.savefig('fifth_sound_43hz.png', dpi=1200, bbox_inches='tight')
plt.close()
