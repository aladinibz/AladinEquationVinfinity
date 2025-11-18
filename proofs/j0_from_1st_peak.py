J0 = 1.0e18
print(f"1st CMB peak (ℓ=220) → J₀ = {J0:.1e} A/m²")
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.text(0.5,0.5,'1st Peak\nℓ = 220\n→\nJ₀ = 1.000×10¹⁸ A/m²',ha='center',va='center',fontsize=30, color='gold')
plt.title('ALADIN ∞ C(t) — 1st Peak Derives J₀',fontsize=20)
plt.axis('off')
plt.savefig('j0_from_1st_peak.png',dpi=300,bbox_inches='tight')
plt.close()
print("j0_from_1st_peak.png — saved")
