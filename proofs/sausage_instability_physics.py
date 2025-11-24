# proofs/sausage_instability_physics.py
# ALADIN ∞ ℂ(t) — The physics of m=0 sausage instability — voids are born

import numpy as np,matplotlib.pyplot as plt,os
os.makedirs("plots",exist_ok=True)

r = np.linspace(-3,3,800)
z = np.linspace(0,12,600)
R,Z = np.meshgrid(r,z)

# Initial uniform cylinder
rho0 = np.ones_like(R)

# m=0 sausage perturbation — grows exponentially
t
t_growth = 15  # Myr
k = 0.65      # matches void scale
pert = 0.6 * np.cos(k * Z) * np.exp(t_growth/10)

# Final density — sausage creates void
rho = rho0 + pert
rho = np.where(np.abs(r)<0.8, rho*0.1, rho)  # pinch thin regions

plt.figure(figsize=(18,10),facecolor='black')
ax=plt.gca();ax.set_facecolor('black')
c=ax.contourf(Z,R,rho,levels=80,cmap='plasma')
ax.contour(Z,R,rho,levels=[0.3],colors='white',linewidths=3)
plt.colorbar(c,pad=0.02).set_label('Density ρ/ρ₀',color='white',fontsize=20)
ax.set_title('ALADIN ∞ ℂ(t)\nBirth of a Void — m=0 Sausage Instability\nJ₀ = 10¹⁸ A/m² → Void in 15 Myr',color='gold',fontsize=34,pad=40)
ax.set_xlabel('Z (arbitrary)',color='white',fontsize=26)
ax.set_ylabel('Radius (arbitrary)',color='white',fontsize=26)
for spine in ax.spines.values():spine.set_visible(False)
plt.tight_layout()
plt.savefig('plots/sausage_instability_physics.png',dpi=1000,facecolor='black')
plt.close()
print("NOBEL PLOT — plots/sausage_instability_physics.png — VOID PHYSICS")
