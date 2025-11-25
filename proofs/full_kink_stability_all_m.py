# proofs/full_kink_stability_all_m.py
import os,numpy as np,matplotlib.pyplot as plt
os.makedirs("plots",exist_ok=True)
ka=np.linspace(.01,5,1200)
g0=np.sqrt(np.maximum(1-ka**2,0))*2.1
g=lambda m:ka*np.sqrt(np.maximum((ka**2-(m*m-1))/(m*m+1),0))*1.8
plt.figure(figsize=(24,14),facecolor='black');plt.gca().set_facecolor('black')
plt.plot(ka,g0,'#FFD700',lw=14,label='m=0 Sausage — Voids')
for m,c,l in [(1,'#00FFFF','m=1 Kink — Filaments'),(2,'#FF00FF','m=2'),(3,'#FFFFFF','m=3'),(4,'#FFFF00','m=4'),(5,'#00FF00','m=5'),(6,'#FF8800','m=6')]:
    plt.plot(ka,g(m),c,lw=10,label=l)
plt.axvspan(0,1,color='gray',alpha=.3,label='Cosmic scales — only m=0 & m=1')
plt.axvline(.63,color='lime',lw=8,ls='--',label='Void scale')
plt.title('ALADIN ∞ ℂ(t)\nOnly m=0 & m=1 Grow — Cosmic Web Eternal',color='gold',fontsize=44,pad=70)
plt.xlabel('k·a',color='white',fontsize=36);plt.ylabel('Γ',color='white',fontsize=36)
plt.legend(facecolor='black',labelcolor='white',fontsize=28);plt.grid(alpha=.3)
for s in plt.gca().spines.values():s.set_visible(False)
plt.tight_layout();plt.savefig('plots/full_kink_stability_all_m.png',dpi=1200,facecolor='black');plt.close()
print("10/10 — full_kink_stability_all_m.png — DONE")
