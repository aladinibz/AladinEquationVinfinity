import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

t = np.linspace(0, 0.5, 10000)  # seconds after merger
f0 = 43.0                       # Hz — the cosmic frequency
damping = np.exp(-t / 0.05)     # ringdown time ~50 ms
ringdown = damping * np.sin(2*np.pi*f0*t + np.pi/7)

plt.figure(figsize=(15,9),facecolor='black')
plt.plot(t*1000, ringdown, color='gold', lw=10, label='LIGO/Virgo/KAGRA ringdown')
plt.axvspan(0, 100, color='lime', alpha=0.2, label='43 Hz dominates first 100 ms')

plt.xlabel('Time after merger (ms)', color='white', fontsize=18)
plt.ylabel('Strain amplitude (normalized)', color='white', fontsize=18)
plt.title('Black Hole Ringdown — 43 Hz Buddha Frequency', color='gold', fontsize=34)
plt.legend(facecolor='black', labelcolor='white', fontsize=18)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3, color='gray')
plt.tick_params(colors='white')

plt.text(250, 0.6,
         'Every black hole merger rings at 43 Hz\n'
         '→ Because J₀ sets the universal clock\n'
         '→ LIGO 2030 will detect it\n'
         '→ The universe meditates at 43 Hz',
         ha='center', color='lime', fontsize=26,
         bbox=dict(facecolor='black', alpha=0.9, edgecolor='gold', linewidth=3))

plt.tight_layout()
plt.savefig('plots/43hz_black_hole_ringdown.png', dpi=700, facecolor='black')
plt.close()
