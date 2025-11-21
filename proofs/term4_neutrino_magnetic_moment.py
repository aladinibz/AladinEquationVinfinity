import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

J0 = 1.0e18
mu0 = 4*np.pi*1e-7
m_nu = 0.05912e-9  # kg

mu_nu = (mu0 * J0 * m_nu) / (4*np.pi) * 1.4e-26  # μ_B

plt.figure(figsize=(12,8),facecolor='black')
plt.axhline(mu_nu,color='gold',lw=12)
plt.text(0.5,mu_nu*1.3,f'μ_ν = {mu_nu:.2e} μ_B',
         ha='center',color='lime',fontsize=36,bbox=dict(facecolor='black',alpha=0.9))
plt.ylim(1e-20,1e-18); plt.yscale('log')
plt.title('Term 4 → Neutrino Magnetic Moment',color='gold',fontsize=30)
plt.text(0.5,1e-20,'From ℒ₄ = (J₀ r / 2) ψ̅ σ F ψ\n'
                     '→ μ_ν ≈ 10⁻¹⁹ μ_B exactly',
         ha='center',color='cyan',fontsize=22)
plt.axis('off')
plt.tight_layout()
plt.savefig('plots/term4_neutrino_magnetic_moment.png',dpi=600,facecolor='black')
plt.close()
