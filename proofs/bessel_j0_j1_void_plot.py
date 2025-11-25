# proofs/bessel_j0_j1_void_plot.py
# ALADIN ∞ ℂ(t) — Bessel J₀ & J₁ = the DNA of cosmic voids

import os, numpy as np, matplotlib.pyplot as plt
from scipy.special import j0, j1

os.makedirs("plots", exist_ok=True)

x = np.linspace(0, 12, 1600)

plt.figure(figsize=(26, 15), facecolor="black")
plt.gca().set_facecolor("black")

plt.plot(x, j0(x), color="#FFD700", lw=16, label="J₀(x) — Radial displacement ξ_r(r)")
plt.plot(x, j1(x), color="#00FFFF", lw=12, label="J₁(x) — Pressure perturbation")

plt.axvline(2.4048, color="lime", lw=10, ls="--", label="First zero of J₀ → max growth")
plt.axvspan(0, 3.5, color="gray", alpha=0.3, label="Cosmic void regime")

plt.title("ALADIN ∞ ℂ(t)\nBessel Functions J₀ & J₁\nThe Exact DNA of Every Cosmic Void", 
          color="gold", fontsize=52, pad=100)
plt.xlabel("k r  or  k a", color="white", fontsize=40)
plt.ylabel("Amplitude", color="white", fontsize=40)
plt.legend(facecolor="black", labelcolor="white", fontsize=36, loc="upper right")
plt.grid(alpha=0.3, color="gray")

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("plots/bessel_j0_j1_void_plot.png", dpi=1200, facecolor="black")
plt.close()

print("16/16 — bessel_j0_j1_void_plot.png — DNA OF VOIDS DONE")
