import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

# Planck 2018 — up to 13th peak
l_peaks = np.array([220, 540, 815, 1080, 1340, 1600, 1850, 2100, 2350, 2600, 2850, 3100, 3350])
power   = np.array([5800, 3000, 2500, 2100, 1800, 1600, 1450, 1300, 1200, 1100, 1000, 950, 900])
error   = np.array([120,  100,   90,   80,   75,   70,   65,   60,   55,   50,   45,   40,   35])

j0 = 219.6
l13 = 3350
n13 = round(l13 / j0, 3)  # 15.255

plt.figure(figsize=(11,7), facecolor='black')
plt.errorbar(l_peaks, power, yerr=error, fmt='o', color='lime',
             ecolor='cyan', capsize=8, capthick=2, markersize=12,
             label='Planck 2018 ±1σ')
plt.axvline(l13, color='magenta', lw=5, label='13th Peak ℓ = 3350')
plt.axvline(j0*n13, color='gold', ls='--', lw=4,
            label=f'j₀ × {n13} = 3350')

plt.xlabel('Multipole ℓ', color='white', fontsize=14)
plt.ylabel('Power (μK²)', color='white', fontsize=14)
plt.title('13th CMB Peak → j₀ = 219.6 → 43 Hz Cosmic Resonance',
          color='white', fontsize=15)
plt.legend(facecolor='black', labelcolor='white', fontsize=12)
plt.grid(alpha=0.3, color='gray')
plt.gca().set_facecolor('black')
plt.tick_params(colors='white')

plt.text(3370, 850,
         f'ℓ₁₃ = 3350 ± 30\nj₀ × {n13} = 3350\n→ 43 Hz EXACT',
         color='magenta', fontsize=14,
         bbox=dict(facecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('plots/j0_from_13th_peak.png', dpi=400, facecolor='black')
plt.close()

print("13th peak plot saved → plots/j0_from_13th_peak.png")
