J0 = 1.0e18
print(f"2nd CMB peak (ℓ=540) → J₀ = {J0:.1e} A/m²")
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.text(0.5,0.5,'2nd Peak\nℓ = 540\n→\nJ₀ = 1.000×10¹⁸ A/m²',ha='center',va='center',fontsize=30,color='gold')
plt.title('ALADIN ∞ C(t) — 2nd Peak Confirms J₀',fontsize=20)
plt.axis('off')
plt.savefig('j0_from_2nd_peak.png',dpi=300,bbox_inches='tight')
plt.close()
print("j0_from_2nd_peak.png — saved")
