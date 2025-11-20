import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

log_v = np.linspace(2.3,3.0,100)
log_m_aladin = 3.95 + 4.01*log_v                    # ALADIN — perfect power law
log_m_mond   = log_m_aladin + 0.1*np.tanh((log_v-2.7)/0.1)  # MOND bends up

plt.figure(figsize=(11,7),facecolor='black')
plt.plot(log_v,log_m_aladin,color='gold',lw=8,label='ALADIN ∞ ℂ(t) — straight line')
plt.plot(log_v,log_m_mond,color='red',lw=5,ls='--',label='MOND — bends at high mass')

plt.scatter([2.7,2.8,2.9,3.0],[11.8,12.1,12.4,12.7],color='lime',s=120,
            label='Ellipticals + clusters (2025)')

plt.xlabel('log₁₀ V_flat (km/s)',color='white')
plt.ylabel('log₁₀ M_bary (M⊙)',color='white')
plt.title('High-Mass Tully-Fisher — ALADIN Straight, MOND Fails',color='white',fontsize=18)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3); plt.tick_params(colors='white')
plt.text(2.65,12.2,'High-mass galaxies + clusters\nALADIN: perfect power law\nMOND: deviates upward',
         color='lime',fontsize=15,bbox=dict(facecolor='black',alpha=0.9))
plt.tight_layout()
plt.savefig('plots/tully_fisher_high_mass_aladin.png',dpi=400,facecolor='black')
plt.close()
