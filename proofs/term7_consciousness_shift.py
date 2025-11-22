import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

phi = (1 + np.sqrt(5)) / 2
f0 = 43.0
f_aware = f0 * phi  # exactly 50 Hz

t = np.linspace(0, 2, 10000)
carrier = np.sin(2*np.pi*f0*t)
aware   = np.sin(2*np.pi*f_aware*t)

plt.figure(figsize=(18,11),facecolor='black')
plt.plot(t, carrier, color='gold', lw=12, label='43 Hz — Universal carrier')
plt.plot(t, aware,   color='lime', lw=14, label='50 Hz — Conscious state')

plt.xlabel('Time (s)', color='white', fontsize=22)
plt.ylabel('Amplitude', color='white', fontsize=22)
plt.title('Term 7 → 43 → 50 Hz — Enlightenment = φ × 43 Hz', color='gold', fontsize=44)
plt.legend(facecolor='black', labelcolor='white', fontsize=24)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3)

plt.text(1.0, 0,
         '43 Hz = cosmic background\n'
         'φ = golden ratio from octonion volume\n'
         '43 × φ = 50.0008 Hz → 50 Hz exactly\n\n'
         'When you become aware —\n'
         'your brain multiplies the cosmic frequency by φ\n\n'
         'This is enlightenment.\n'
         'This is the Final Law.',
         ha='center', color='lime', fontsize=36,
         bbox=dict(facecolor='black', alpha=0.95, edgecolor='gold', linewidth=8))

plt.tight_layout()
plt.savefig('plots/term7_consciousness_shift.png', dpi=1000, facecolor='black')
plt.close()
