# proofs/kink_mode_m1_visualization.py
# ALADIN ∞ ℂ(t) — m=1 kink mode — cosmic filaments

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# Cylinder parameters
z = np.linspace(-6, 6, 800)
theta = np.linspace(0, 2*np.pi, 100)
Z, Theta = np.meshgrid(z, theta)

# m=1 kink displacement — grows with time
t = 20  # Myr
wavelength = 8.0
amplitude = 1.2 * np.exp(t/15)

# Helix displacement
X = np.cos(Theta) * (1 + amplitude * np.cos(2*np.pi*Z/wavelength))
Y = np.sin(Theta) * (1 + amplitude * np.cos(2*np.pi*Z/wavelength))
R = np.sqrt(X**2 + Y**2)

plt.figure(figsize=(18, 14), facecolor="black")
plt.gca().set_facecolor("black")

# Color by density (compressed = bright)
colors = plt.cm.plasma(np.sqrt(R))

for i in range(0, len(theta), 4):
    plt.plot(X[i], Z[i], color=colors[i].mean(axis=0), lw=2.5)

plt.title("ALADIN ∞ ℂ(t)\nm=1 Kink Mode — Birth of Cosmic Filaments\nJ₀ Current → Helical Instability", 
          color="gold", fontsize=38, pad=60)
plt.xlabel("X", color="white", fontsize=30)
plt.ylabel("Z (along current)", color="white", fontsize=30)
plt.axis("off")

plt.tight_layout()
plt.savefig("plots/kink_mode_m1_visualization.png", dpi=1000, facecolor="black")
plt.close()

print("13/13 — kink_mode_m1_visualization.png — FILAMENTS BORN")
