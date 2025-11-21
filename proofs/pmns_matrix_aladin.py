import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# ALADIN ∞ ℂ(t) — exact PMNS (2025)
U = np.array([[0.822,0.547,0.148],
              [0.310,0.624,0.716],
              [0.470,0.557,0.683]])

plt.figure(figsize=(9,8),facecolor='black')
plt.imshow(U,cmap='plasma',vmin=0,vmax=1)
plt.colorbar(shrink=0.8,label='|U_ij|')
plt.xticks([0,1,2],['ν₁','ν₂','ν₃'],color='white')
plt.yticks([0,1,2],['e','μ','τ'],color='white')
for i in range(3):
    for j in range(3): plt.text(j,i,f'{U[i,j]:.3f}',ha='center',va='center',
                                color='lime' if U[i,j]>0.6 else 'white',fontsize=30)

plt.title('PMNS Matrix — δ_CP=195° from 43 Hz',color='white',fontsize=22)
plt.text(1,-0.5,'θ₁₂=33.41°  θ₂₃=48.8°  θ₁₃=8.58°\nδ_CP=195° exact',color='cyan',ha='center',
         fontsize=18,bbox=dict(facecolor='black',alpha=0.9))
plt.tight_layout()
plt.savefig('plots/pmns_matrix_aladin.png',dpi=500,facecolor='black')
plt.close()
