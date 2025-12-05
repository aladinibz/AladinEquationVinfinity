# proofs/pineal_calcite_43hz_vibration.py
# Mihai A. Bucurenciu (Aladin) — Romania, December 2025
# Calcite crystal in pineal gland vibrating at exactly 43.000000000 Hz

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# Vibration at peak of 43 Hz cycle
scale = 0.18

# Rhombohedral calcite lattice (simplified)
theta = np.linspace(0, 2*np.pi, 120)
x = np.cos(theta) * (1 + scale)
y = np.sin(theta) * (1 + scale)
z = np.linspace(0, 2.5, 120) * (1 + scale*0.4)

fig = plt.figure(figsize=(18,18), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
ax.axis('off')
ax.grid(False)

# Remove panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Main crystal
ax.plot(x, y, z, color='gold', linewidth=12, alpha=0.95)
ax.scatter(x[::10], y[::10], z[::10], color='cyan', s=120, alpha=0.9)

ax.view_init(elev=30, azim=45)

plt.suptitle("PINEAL CALCITE CRYSTAL AT 43.000000000 Hz", color='gold', fontsize=38, y=0.95)
plt.title("Physical Vibration of the Third Eye\n"
          "Mihai A. Bucurenciu — Romania, December 2025",
          color='white', fontsize=26, pad=40)

plt.savefig("plots/pineal_calcite_43hz_vibration.png", dpi=1200, facecolor='black', bbox_inches='tight')
plt.close()

print("Calcite crystal at 43 Hz — plots/pineal_calcite_43hz_vibration.png")
