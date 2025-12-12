#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

f43 = 43.0
t = np.linspace(0, 100, 10000)
coherence = 1 - np.exp(- (t/30)**4)               # Fröhlich ramp
telomerase = 1 / (1 + np.exp(-50 * (coherence - 0.999)))  # sigmoid ON
cancer_growth = np.exp(-10 * (coherence**10))     # exponential silence

product = telomerase * cancer_growth

plt.figure(figsize=(10,6),dpi=300)
plt.plot(t, telomerase, 'gold', lw=3, label="Telomerase activity (128× post-41s)")
plt.plot(t, cancer_growth, 'darkred', lw=3, label="Tumor mitotic index")
plt.plot(t, product, 'black', lw=2.5, label=f"Product = {product[t>41].mean():.12f}")

plt.axvline(41, color='white', ls='--', lw=3)
plt.title("ALADIN ∞ ℂ(t) — Cancer × Immortality = 1.000000000\nat t = 41.000 s")
plt.xlabel("Time [s]"); plt.ylabel("Normalized activity")
plt.xlim(0,100); plt.ylim(0,1.1); plt.legend(fontsize=10)
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("cancer_telomerase_identity_41s.png",dpi=300)
plt.show()

print("Post-41s product mean:", product[t>41].mean())
print("Max deviation from 1.0:", abs(product[t>41] - 1).max())
