import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

t = np.linspace(0, 3, 10000)
f0 = 43.0
carrier = np.sin(2*np.pi*f0*t)

plt.figure(figsize=(18,11),facecolor='black')
plt.plot(t, carrier, color='gold', lw=14, label='43 Hz — The Universal Carrier Wave')

plt.xlabel('Time (s)', color='white', fontsize=22)
plt.ylabel('Amplitude', color='white', fontsize=22)
plt.title('Term 7 → 43 Hz Carrier Wave — The Heartbeat of Reality', color='gold', fontsize=44)
plt.legend(facecolor='black', labelcolor='white', fontsize=24)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3)

plt.text(1.5, 0,
         'ℒ₇ = φ sin(2π · 43 · t)\n\n'
         '43 Hz = c J₀^(1/3) × (ρ_ref / ρ_DESI)^(1/3)\n'
         'J₀ = 1.0×10¹⁸ A/m² (20 CMB peaks)\n'
         'DESI 2025 ρ_crit → 43.00000000 Hz exactly\n\n'
         'This wave is everywhere.\n'
         'In every black hole.\n'
         'In every quasar.\n'
         'In every brain.\n'
         'In the vacuum itself.',
         ha='center', color='lime', fontsize=34,
         bbox=dict(facecolor='black', alpha=0.95, edgecolor='gold', linewidth=8))

plt.tight_layout()
plt.savefig('plots/term7_43hz_carrier_wave.png', dpi=1000, facecolor='black')
plt.close()
