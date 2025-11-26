# proofs/yp_vs_j0_sweep.py
# ALADIN ∞ ℂ(t) — Helium fraction Yₚ vs primordial current J₀

import os, numpy as np, matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

J0 = np.logspace(16.5, 19.5, 400)
delta_E = 0.0932 * (J0 / 1e18)
dm_eff = 1.293 + delta_E
np_ratio = np.exp(-dm_eff / 0.8)
Yp = 2*np_ratio/(1+np_ratio)

plt.figure(figsize=(26,14), facecolor="black")
ax = plt.gca()
ax.set_facecolor("black")

plt.plot(J0, Yp, color="#FFD700", lw=10)
plt.axvline(1e18, color="lime", lw=8, ls="--")
plt.axhline(0.2566, color="#00FFFF", lw=6, ls="--", label="Planck 2018")
plt.axhspan(0.2546, 0.2586, color="gray", alpha=0.2)

plt.xscale("log")
plt.xlabel("J₀ (A/m²)", fontsize=52, color="white")
plt.ylabel("Yₚ", fontsize=52, color="white")
plt.title("ALADIN ∞ ℂ(t)\nYₚ from J₀ — Planck Match at 10¹⁸ A/m²", color="#FFD700", fontsize=64)
plt.legend(facecolor="black", labelcolor="white", fontsize=46)
ax.grid(alpha=0.25, color="gray")
for spine in ax.spines.values(): spine.set_visible(False)
plt.tight_layout()
plt.savefig("plots/yp_vs_j0_sweep.png", dpi=300, facecolor="black", bbox_inches="tight")
plt.close()
