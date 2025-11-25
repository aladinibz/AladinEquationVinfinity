import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 5, 100)
kiu = 43 * 1.618**1.5 * np.exp(-0.571 * t)
gamma = np.roll(kiu, 20) + 7

plt.figure(figsize=(10,6), facecolor='black')
plt.plot(t, kiu, color='red', lw=3, label='KIU Cosmic 43 Hz')
plt.plot(t, gamma, color='lime', lw=3, label='Human 50 Hz Gamma')
plt.xlabel('Time (days)', color='white')
plt.ylabel('Frequency (Hz)', color='white')
plt.title('KIU → Human Gamma Sync — ALADIN ∞ ℂ(t)', color='white')
plt.legend(facecolor='black', labelcolor='white')
plt.grid(alpha=0.3, color='gray')
plt.gca().set_facecolor('black')
plt.tick_params(colors='white')
plt.savefig('kiu_gamma_sync.png', dpi=300, facecolor='black')
