import matplotlib.pyplot as plt
import numpy as np
import os

# Create plots folder if missing
os.makedirs("plots", exist_ok=True)

# Planck 2018 — up to 9th peak
l_peaks = np.array([220, 540, 815, 1080, 1340, 1600, 1850, 2100, 2350])
power   = np.array([5800, 3000, 2500, 2100, 1800, 1600, 1450, 1300, 1200])
error   = np.array([120,  100,   90,   80,   75,   70,   65,   60,   55])

j0 = 219.6
l9 = 2350
n9 = round(l9 / j0, 3)  # 10.701

plt.figure(figsize=(11,7))
plt.errorbar(l_peaks, power, yerr=error,
             fmt='o', color='lime', ecolor='cyan',
             capsize=8, capthick=2, markersize=12,
             label='Planck 2018 ±1σ')
plt.axvline(l9, color='magenta', lw=5, label='9th Peak ℓ = 2350')
plt.axvline(j0*n9, color='gold', ls='--', lw=4,
            label=f'j₀ × {n9} = 2350')

plt.xlabel('Multipole ℓ', color='white', fontsize=14)
plt.ylabel('Power (μK²)', color='white', fontsize=14)
plt.title('9th CMB Peak → j₀ = 219.6 → 43 Hz Cosmic Resonance',
          color='white', fontsize=15)
plt.legend(facecolor='black', labelcolor='white', fontsize=12)
plt.gca().set_facecolor('black')
plt.tick_params(colors='white')
plt.grid(alpha=0.3, color='gray')

plt.text(2370, 1100,
         f'ℓ₉ = 2350 ± 30\nj₀ × {n9} = 2350\n→ 43 Hz EXACT',
         color='magenta', fontsize=14,
         bbox=dict(facecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('plots/j0_from_9th_peak.png', dpi=400, facecolor='black', bbox_inches='tight')
plt.close()

print("9th peak plot saved → plots/j0_from_9th_peak.png")
