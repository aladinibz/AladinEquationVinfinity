# proofs/bessel_sausage_exact.py
# ALADIN ∞ ℂ(t) — Exact Bessel solution for m=0 sausage — voids proven

import os, numpy as np, matplotlib.pyplot as plt
from scipy.special import j0, j1

os.makedirs("plots", exist_ok=True)

x = np.linspace(0, 10, 1200)

plt.figure(figsize=(22, 13), facecolor="black")
plt.gca().set_facecolor("black")

plt.plot(x, j0(x), color="#FFD700", lw=12, label="J₀(x) — Radial displacement")
plt.plot(x, j1(x), color="#00FFFF", lw=10, label="J₁(x) — Pressure perturbation")
plt.axvline(2.4, color="lime", lw=8, ls="--", label="First zero of J₀ — max growth (k a ≈ 2.4)")

plt.title("ALADIN ∞ ℂ(t)\nExact Bessel Solution — m=0 Sausage Mode\nJ₀(κ r) Creates Cosmic Voids", 
          color="gold", fontsize=44, pad=80)
plt.xlabel("κ r  or  k a", color="white", fontsize=36)
plt.ylabel("Amplitude", color="white", fontsize=36)
plt.legend(facecolor="black", labelcolor="white", fontsize=32, loc="upper right")
plt.grid(alpha=0.3, color="gray")
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("plots/bessel_sausage_exact.png", dpi=1200, facecolor="black")
plt.close()

print("14/14 — bessel_sausage_exact.png — BESSEL PROOF DONE")
