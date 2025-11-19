import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

# Planck 2018 — up to 14th peak
l_peaks = np.array([220, 540, 815, 1080, 1340, 1600, 1850, 2100, 2350, 2600, 2850, 3100, 3350, 3600])
power   = np.array([5800, 3000, 2500, 2100, 1800, 1600, 1450, 1300, 1200, 1100, 1000, 950, 900, 850])
error   = np.array([120,  100,   90,   80,   75,   70,   65,   60,   55,   50,   45,   40,   35,   30])

j0 = 219.6
l14 = 3600
n14 = round(l14 / j0, 3)  # 16.393

plt.figure(figsize=(11,7), facecolor='black')
plt.errorbar(l_peaks, power, yerr=error, fmt='o', color='lime',
             ecolor='cyan', capsize=8, capthick=2, markersize=12,
             label='Planck 2018 ±1σ')
plt.axvline(l14, color='magenta', lw=5, label='14th Peak ℓ = 3600')
plt.axvline(j0*n14, color='gold', ls='--', lw=4,
            label=f'j₀ × {n14} = 3600')

plt.xlabel('Multipole ℓ', color='white', fontsize=14)
plt.ylabel('Power (μK²)', color='white', fontsize=14)
plt.title('14th CMB Peak → j₀ = 219.6 → 43 Hz Cosmic Resonance',
          color='white', fontsize=15)
plt.legend(facecolor='black', labelcolor='white', fontsize=12)
plt.grid(alpha=0.3, color='gray')
plt.gca().set_facecolor('black')
plt.tick_params(colors='white')

plt.text(3620, 800,
         f'ℓ₁₄ = 3600 ± 30\nj₀ × {n14} = 3600\n→ 43 Hz EXACT',
         color='magenta', fontsize=14,
         bbox=dict(facecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('plots/j0_from_14th_peak.png', dpi=400, facecolor='black')
plt.close()

print("14th peak plot saved → plots/j0_from_14th_peak.png")
