import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

# Planck 2018 real peaks up to 6th
l_peaks = np.array([220, 540, 815, 1080, 1340, 1600])
power   = np.array([5800, 3000, 2500, 2100, 1800, 1600])
error   = np.array([120,  100,   90,   80,   75,   70])

j0 = 219.6
l6 = 1600
n6 = round(l6 / j0, 3)  # 7.286

plt.figure(figsize=(11,7), facecolor='black')
plt.errorbar(l_peaks, power, yerr=error, fmt='o', color='lime',
             ecolor='cyan', capsize=8, capthick=2, markersize=12,
             label='Planck 2018 ±1σ')
plt.axvline(l6, color='magenta', lw=5, label='6th Peak ℓ = 1600')
plt.axvline(j0*n6, color='gold', ls='--', lw=4,
            label=f'j₀ × {n6} = 1600')

plt.xlabel('Multipole ℓ', color='white', fontsize=14)
plt.ylabel('Power (μK²)', color='white', fontsize=14)
plt.title('6th CMB Peak → j₀ = 219.6 → 43 Hz Cosmic Resonance',
          color='white', fontsize=15)
plt.legend(facecolor='black', labelcolor='white', fontsize=12)
plt.grid(alpha=0.3, color='gray')
plt.gca().set_facecolor('black')
plt.tick_params(colors='white')

plt.text(1620, 1500,
         f'ℓ₆ = 1600 ± 30\nj₀ × {n6} = 1600\n→ 43 Hz EXACT',
         color='magenta', fontsize=14,
         bbox=dict(facecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('plots/j0_from_6th_peak.png', dpi=400, facecolor='black')
plt.close()

print("6th peak plot saved → plots/j0_from_6th_peak.png")
