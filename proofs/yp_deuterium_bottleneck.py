# proofs/yp_deuterium_bottleneck.py
# ALADIN ∞ ℂ(t) — Final Yₚ after deuterium bottleneck delay

import os, numpy as np, matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# Time delay from deuterium bottleneck (standard correction)
delay = np.linspace(0, 300, 400)  # seconds after T≈0.08 MeV
Yp_raw = 0.2632  # Yₚ right after n/p freeze-out (with J₀ shift)
Yp_final = Yp_raw - 0.0068 * np.exp(-delay/60)  # standard exponential delay

plt.figure(figsize=(26,14), facecolor="black")
ax = plt.gca()
ax.set_facecolor("black")

plt.plot(delay, Yp_final, color="#FF6B00", lw=10)
plt.axhline(0.2566, color="#FFD700", lw=8, ls="--", label="Planck 2018")
plt.axhspan(0.2546, 0.2586, color="gray", alpha=0.2)

plt.xlabel("Time after n/p freeze-out (s)", fontsize=52, color="white")
plt.ylabel("Yₚ", fontsize=52, color="white")
plt.title("ALADIN ∞ ℂ(t)\nYₚ Final Value after Deuterium Bottleneck", color="#FFD700", fontsize=64)
plt.legend(facecolor="black", labelcolor="white", fontsize=46)
ax.grid(alpha=0.25, color="gray")
for spine in ax.spines.values(): spine.set_visible(False)
plt.tight_layout()
plt.savefig("plots/yp_deuterium_bottleneck.png", dpi=300, facecolor="black", bbox_inches="tight")
plt.close()
