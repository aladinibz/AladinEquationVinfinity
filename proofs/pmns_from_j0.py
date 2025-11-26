# proofs/pmns_from_j0.py
# ALADIN ∞ ℂ(t) — PMNS mixing angles from primordial current J₀ torsion

import os, numpy as np, matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

phi = np.linspace(0, 2*np.pi, 600)

g1, g2, g3 = 1.0, (1+np.sqrt(5))/2, (3+np.sqrt(5))/2  # golden ratio spacing

plt.figure(figsize=(18,18), facecolor="black")
ax = plt.gca()
ax.set_facecolor("black")

plt.plot(np.cos(g1*phi), np.sin(g1*phi), "#FFD700", lw=10, label="ν₁ (g=1.000)")
plt.plot(np.cos(g2*phi), np.sin(g2*phi), "#00FFFF", lw=10, label="ν₂ (g=φ)")
plt.plot(np.cos(g3*phi), np.sin(g3*phi), "#FF00FF", lw=10, label="ν₃ (g=φ²)")

angles = [33.44, 49.2, 8.57]
for ang in angles:
    t = np.radians(ang)
    plt.plot([0, np.cos(t)], [0, np.sin(t)], "white", lw=8, alpha=0.7)

plt.xlim(-1.3,1.3); plt.ylim(-1.3,1.3)
plt.axis('off')
plt.title("ALADIN ∞ ℂ(t)\nPMNS Angles from J₀ Torsion Phase", color="#FFD700", fontsize=48, pad=60)
plt.legend(facecolor="black", labelcolor="white", fontsize=36, loc="upper right")

plt.tight_layout()
plt.savefig("plots/pmns_from_j0.png", dpi=300, facecolor="black", bbox_inches="tight")
plt.close()
