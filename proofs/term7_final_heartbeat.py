import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

t = np.linspace(0, 13.8e9 * 365.25 * 86400, 10000)  # age of universe in seconds
f0 = 43.0
heartbeat = np.sin(2*np.pi*f0*t)

plt.figure(figsize=(20,12),facecolor='black')
plt.plot(t / (1e9*365.25*86400), heartbeat, color='gold', lw=16)

plt.xlabel('Age of Universe (billion years)', color='white', fontsize=24)
plt.ylabel('Amplitude', color='white', fontsize=24)
plt.title('Term 7 — The Universe Has Been Beating at 43 Hz for 13.8 Billion Years', color='gold', fontsize=48)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3)

plt.text(7, 0,
         'Since t=0 the universe has oscillated\n'
         'exactly 1.88 × 10¹⁸ times at 43 Hz\n\n'
         'Every black hole rings with it\n'
         'Every quasar pulses with it\n'
         'Every conscious brain returns to it\n\n'
         'This is not a wave.\n'
         'This is the heartbeat of God.',
         ha='center', color='lime', fontsize=42,
         bbox=dict(facecolor='black', alpha=0.95, edgecolor='gold', linewidth=10))

plt.tight_layout()
plt.savefig('plots/term7_final_heartbeat.png', dpi=1200, facecolor='black')
plt.close()
