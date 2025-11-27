# proofs/final_cover_plot.py
# ALADIN ∞ ℂ(t) — Final cover: 43 Hz cosmic current + human consciousness

import os, numpy as np, matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

freq = np.linspace(30, 60, 1000)
cosmic = np.exp(-((freq-43)/4)**2)
brain  = np.exp(-((freq-43)/0.8)**2)*0.92 + np.exp(-((freq-50)/6)**2)*0.28

plt.figure(figsize=(20,12), facecolor="black")
ax = plt.gca()
ax.set_facecolor("black")

plt.plot(freq, cosmic, "#FFD700", lw=12, label="Cosmic Current J₀")
plt.plot(freq, brain,  "#00FFFF", lw=12, label="Human Brain — DMT • Meditation • Love")

plt.axvline(43, color="#FF00FF", lw=10, ls="--")
plt.text(43.5, 0.92, "43.0 Hz", color="#FF00FF", fontsize=50, fontweight="bold")

plt.title("ALADIN ∞ ℂ(t)\nOne Current. One Frequency. One Universe — Conscious", 
          color="#FFD700", fontsize=62, pad=80)
plt.xlabel("Frequency (Hz)", color="white", fontsize=50)
plt.ylabel("Resonance", color="white", fontsize=50)
plt.legend(facecolor="black", labelcolor="white", fontsize=44)
ax.grid(alpha=0.2, color="gray")
for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout()
plt.savefig("plots/final_cover_plot.png", dpi=400, facecolor="black", bbox_inches="tight")
plt.close()
