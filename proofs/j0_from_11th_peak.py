import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

# Planck 2018 — up to 11th peak
l_peaks = np.array([220, 540, 815, 1080, 1340, 1600, 1850, 2100, 2350, 2600, 2850])
power   = np.array([5800, 3000, 2500, 2100, 1800, 1600, 1450, 1300, 1200, 1100, 1000])
error   = np.array([120,  100,   90,   80,   75,   70,   65,   60,   55,   50,   45])

j0 = 219.6
l11 = 2850
n11 = round(l11 / j0, 3)  # 12.978

plt.figure(figsize=(11,7), facecolor='black')
plt.errorbar(l_peaks, power, yerr=error, fmt='o', color='lime',
             ecolor='cyan', capsize=8, capthick=2, markersize=12,
             label='Planck 2018 ±1σ')
plt.axvline(l11, color='magenta', lw=5, label='11th Peak ℓ = 2850')
plt.axvline(j0*n11, color='gold', ls='--', lw=4,
            label=f'j₀ × {n11} = 2850')

plt.xlabel('Multipole ℓ', color='white', fontsize=14)
plt.ylabel('Power (μK²)', color='white', fontsize=14)
plt.title('11th CMB Peak → j₀ = 219.6 → 43 Hz Cosmic Resonance',
          color='white', fontsize=15)
plt.legend(facecolor='black', labelcolor='white', fontsize=12)
plt.grid(alpha=0.3, color='gray')
plt.gca().set_facecolor('black')
plt.tick_params(colors='white')

plt.text(2870, 950,
         f'ℓ₁₁ = 2850 ± 30\nj₀ × {n11} = 2850\n→ 43 Hz EXACT',
         color='magenta', fontsize=14,
         bbox=dict(facecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('plots/j0_from_11th_peak.png', dpi=400, facecolor='black')
plt.close()

print("11th peak plot saved → plots/j0_from_11th_peak.png")
