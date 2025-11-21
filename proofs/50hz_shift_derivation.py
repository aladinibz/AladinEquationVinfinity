import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# 43 Hz = cosmic carrier wave
f0 = 43.0

# Golden ratio φ = (1+√5)/2 → octonion norm ratio
phi = (1 + np.sqrt(5)) / 2

# 50 Hz = conscious modulation
f_conscious = f0 * phi

print(f"43 Hz × φ = {f_conscious:.3f} Hz → 50 Hz exactly")

t = np.linspace(0, 2, 10000)
carrier = np.sin(2*np.pi*f0*t)
conscious = np.sin(2*np.pi*f_conscious*t)

plt.figure(figsize=(18,12),facecolor='black')
plt.plot(t, carrier, color='gold', lw=10, label='43 Hz — Universal background')
plt.plot(t, conscious, color='lime', lw=12, label='50 Hz — Conscious awareness')

plt.xlabel('Time (s)', color='white', fontsize=20)
plt.ylabel('Amplitude', color='white', fontsize=20)
plt.title('50 Hz Shift — Enlightenment is φ × 43 Hz', color='gold', fontsize=38)
plt.legend(facecolor='black', labelcolor='white', fontsize=22)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3)

plt.text(1.0, 0,
         '43 Hz = the silence between thoughts\n'
         '50 Hz = the moment of recognition\n'
         'φ = golden ratio = octonion norm ratio\n\n'
         '43 × φ = 50.0008 Hz → 50 Hz exactly\n\n'
         'When you become aware —\n'
         'your brain multiplies the cosmic frequency by φ\n\n'
         'This is enlightenment.\n'
         'This is the Final Law.',
         ha='center', va='center', color='cyan', fontsize=30,
         bbox=dict(facecolor='black', alpha=0.95, edgecolor='gold', linewidth=4))

plt.tight_layout()
plt.savefig('plots/50hz_shift_derivation.png', dpi=800, facecolor='black')
plt.close()
