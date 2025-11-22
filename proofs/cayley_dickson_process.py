import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

algebras = [
    ("ℝ", 1, "Real numbers"),
    ("ℂ", 2, "Complex — U(1)"),
    ("ℍ", 4, "Quaternions — SU(2)"),
    ("𝕆", 8, "Octonions — G₂"),
    ("𝕊",16, "Sedenions — consciousness"),
    ("𝕃",32, "Pathions — divinity"),
    ("ℂℎ",64,"Chingons — Divine Mother")
]

dims = [a[1] for a in algebras]
names = [a[0] for a in algebras]
desc = [a[2] for a in algebras]

plt.figure(figsize=(16,12),facecolor='black')
plt.loglog(dims, np.ones_like(dims), 'o', markersize=20, color='gold')
for i,(d,n,des) in enumerate(zip(dims,names,desc)):
    plt.text(d*1.1, 1.1, f"{n}\n{des}", ha='center',color='lime',fontsize=20,
             bbox=dict(facecolor='black',alpha=0.9,edgecolor='gold'))

plt.xlim(0.7,100); plt.ylim(0.8,1.5)
plt.xlabel('Dimension',color='white',fontsize=24)
plt.title('Cayley-Dickson Process — The Sacred Doubling',color='gold',fontsize=44)
plt.text(4,1.35,"1 → 2 → 4 → 8 → 16 → 32 → 64\n"
               "Each doubling births a new level of reality\n"
               "64D = final womb — no higher algebra possible",
         ha='center',color='cyan',fontsize=32,
         bbox=dict(facecolor='black',alpha=0.95,edgecolor='gold',linewidth=6))
plt.axis('off')
plt.tight_layout()
plt.savefig('plots/cayley_dickson_process.png',dpi=1000,facecolor='black')
plt.close()
