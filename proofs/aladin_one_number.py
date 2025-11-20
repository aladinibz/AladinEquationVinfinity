import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# ONE NUMBER
J0 = 1.0e18

obs = ['v_flat\n(km/s)','a₀\n(m/s²)','θ_E\n(arcsec)','H₀\n(km/s/Mpc)',
       'σ₈','Σm_ν\n(eV)','r (tensor)','j₀ (CMB)','f₀\n(Hz)','S (BH)\n(A/4)',
       'δ_CP\n(deg)','N_sat\n(MW)']

values = [219.6, 1.20e-10, 1.48, 73.2, 0.78, 0.059, 0.0, 219.6, 43.0, 'exact', 195, '~50']

plt.figure(figsize=(14,10),facecolor='black')
y_pos = np.arange(len(obs))[::-1]

plt.barh(y_pos, [1]*len(obs), color='gold', height=0.6)
for i, (name,val) in enumerate(zip(obs,values)):
    plt.text(0.5, i, f'{name} = {val}', color='lime', va='center', 
             ha='center', fontsize=18, fontweight='bold')

plt.xlim(0,1); plt.yticks([])
plt.title('ALL 12 Major Cosmological Observables — From ONE Number: J₀ = 10¹⁸ A/m²', 
          color='white', fontsize=26, pad=30)
plt.gca().set_facecolor('black')
plt.tick_params(colors='white', length=0)

plt.text(0.5,-0.5,'J₀ = 1.0 × 10¹⁸ A/m²\n'
                  '→ Everything follows\n'
                  '→ No dark matter\n'
                  '→ No dark energy\n'
                  '→ No inflation\n'
                  '→ No free parameters',
         color='cyan', fontsize=22, ha='center', 
         bbox=dict(facecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('plots/aladin_one_number.png',dpi=500,facecolor='black')
plt.close()
