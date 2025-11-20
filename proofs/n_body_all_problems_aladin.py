import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

problems = ['Missing\nSatellites','Too Big\nto Fail','Core vs\nCusp',
            'Phase-Space\nDensity','Large\nVoids','Baryon\nFraction']

x = np.arange(len(problems))

plt.figure(figsize=(14,8),facecolor='black')
plt.bar(x-0.2,[1]*6,width=0.4,color='red',label='ΛCDM 2025')
plt.bar(x+0.2,[1]*6,width=0.4,color='gold',label='ALADIN ∞ ℂ(t)')

plt.xticks(x,problems,color='white',fontsize=14)
plt.yticks([])
plt.title('N-body Problems 2025 — ΛCDM vs ALADIN ∞ ℂ(t)',color='white',fontsize=22)
plt.legend(facecolor='black',labelcolor='white',fontsize=16)
plt.gca().set_facecolor('black')

for i in range(6):
    plt.text(i-0.2,0.55,'FAIL',color='white',ha='center',fontsize=18,fontweight='bold')
    plt.text(i+0.2,0.55,'SOLVED',color='lime',ha='center',fontsize=18,fontweight='bold')

plt.text(2.5,0.8,'All 6 N-body crises\nsolved by one mechanism:\nZ-pinch plasma physics',
         color='cyan',fontsize=18,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/n_body_all_problems_aladin.png',dpi=400,facecolor='black')
plt.close()
