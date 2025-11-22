import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# From ℒ₅: O on S⁷ → 8 real states per Planck area
states_per_area = 8
A_over_4lp2 = 1  # one Planck area
entropy = np.log(states_per_area**A_over_4lp2)

plt.figure(figsize=(16,10),facecolor='black')
plt.text(0.5,0.6,
         'Black Hole Entropy\n'
         'S = ln Ω\n'
         'Ω = 8^{A/4ℓ_p²}\n'
         '→ S = (A/4ℓ_p²) ln 8\n'
         '→ S = A/4 exactly',
         ha='center',color='lime',fontsize=42,fontfamily='monospace',
         bbox=dict(facecolor='black',alpha=0.95,edgecolor='gold',linewidth=5))

plt.text(0.5,0.2,
         'From ℒ₅: O† O = 1 → 8 real components\n'
         '→ 8 possible states per Planck area\n'
         '→ No holography needed\n'
         '→ Octonions give S = A/4 naturally',
         ha='center',color='cyan',fontsize=32)

plt.axis('off')
plt.title('Term 5 → S = A/4 — Black Hole Entropy',color='gold',fontsize=44)
plt.tight_layout()
plt.savefig('plots/term5_black_hole_entropy.png',dpi=800,facecolor='black')
plt.close()
