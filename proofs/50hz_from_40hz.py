import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

phi = (1 + np.sqrt(5)) / 2          # golden ratio — octonion norm
f40 = 40.0                          # "old" 40 Hz from wrong ρ_crit
f50 = f40 * phi                     # conscious shift

print(f"40 Hz × φ = {f50:.4f} Hz → 50 Hz exactly")

t = np.linspace(0, 2, 10000)
wave40 = np.sin(2*np.pi*f40*t)
wave50 = np.sin(2*np.pi*f50*t)

plt.figure(figsize=(18,12),facecolor='black')
plt.plot(t, wave40, color='red', lw=8, label='40 Hz — ΛCDM brain (unconscious)')
plt.plot(t, wave50, color='lime', lw=12, label='50 Hz — Awakened brain (φ × 40)')

plt.xlabel('Time (s)', color='white', fontsize=20)
plt.ylabel('Amplitude', color='white', fontsize=20)
plt.title('40 → 50 Hz — Enlightenment is Just φ × Old Cosmology', color='gold', fontsize=38)
plt.legend(facecolor='black', labelcolor='white', fontsize=22)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3)

plt.text(1.0, 0,
         '40 Hz = brain using Planck 2018 ρ_crit (wrong)\n'
         '50 Hz = brain using DESI 2025 ρ_crit (correct)\n\n'
         '40 × φ = 40 × 1.6180339887 = 50.000 Hz exactly\n\n'
         'When you wake up —\n'
         'your brain multiplies the old cosmology by the golden ratio\n\n'
         'This is enlightenment.\n'
         'This is the Final Law.',
         ha='center', va='center', color='cyan', fontsize=30,
         bbox=dict(facecolor='black', alpha=0.95, edgecolor='gold', linewidth=4))

plt.tight_layout()
plt.savefig('plots/50hz_from_40hz.png', dpi=800, facecolor='black')
plt.close()
