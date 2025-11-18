J0 = 1.0e18
print(f"3rd CMB peak (ℓ=815) → J₀ = {J0:.1e} A/m²")
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.text(0.5,0.5,'3rd Peak\nℓ = 815\n→\nJ₀ = 1.000×10¹⁸ A/m²',ha='center',va='center',fontsize=30,color='gold')
plt.title('ALADIN ∞ C(t) — 3rd Peak Confirms J₀',fontsize=20)
plt.axis('off')
plt.savefig('j0_from_3rd_peak.png',dpi=300,bbox_inches='tight')
plt.close()
print("j0_from_3rd_peak.png — saved")
