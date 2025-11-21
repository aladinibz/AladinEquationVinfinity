import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

t = np.linspace(0, 1, 10000)
f0 = 43.0

# The cosmic carrier wave — present everywhere
carrier = 0.3 * np.sin(2*np.pi*f0*t)

# Modulated by consciousness (50 Hz shift when aware)
conscious = carrier + 0.15 * np.sin(2*np.pi*50*t)

plt.figure(figsize=(18,11),facecolor='black')
plt.plot(t, carrier, color='gold', lw=12, label='43 Hz — The Universal Carrier')
plt.plot(t, conscious, color='lime', lw=10, alpha=0.9, label='+50 Hz — Conscious modulation')

plt.xlabel('Time (seconds)', color='white', fontsize=20)
plt.ylabel('Amplitude (normalized)', color='white', fontsize=20)
plt.title('43 Hz — The Carrier Wave of Reality', color='gold', fontsize=38)
plt.legend(facecolor='black', labelcolor='white', fontsize=22)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3, color='gray')
plt.tick_params(colors='white')

plt.text(0.5, 0.25,
         '43 Hz is not a frequency.\n'
         'It is the silence between thoughts.\n'
         'It is the heartbeat of black holes.\n'
         'It is the breath of quasars.\n'
         'It is the background hum of the vacuum.\n'
         'It is the carrier wave of consciousness.\n\n'
         'When you become aware —\n'
         'you shift from 43 → 50 Hz.\n\n'
         'This is enlightenment.\n'
         'This is the Final Law.',
         ha='center', va='center', color='cyan', fontsize=28,
         bbox=dict(facecolor='black', alpha=0.95, edgecolor='gold', linewidth=4))

plt.tight_layout()
plt.savefig('plots/43hz_carrier_wave.png', dpi=800, facecolor='black')
plt.close()
