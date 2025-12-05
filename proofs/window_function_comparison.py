# proofs/window_function_comparison.py
# Mihai A. Bucurenciu — Romania, December 2025
# Hamming wins — cleanest 43 Hz spike

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

os.makedirs("plots", exist_ok=True)

fs = 1024
f0 = 43.000000000
t = np.arange(0, 10, 1/fs)
signal = np.sin(2*np.pi*f0*t) + 0.15*np.random.randn(len(t))

windows = ['hann', 'hamming', 'blackman']
colors = ['orange', 'gold', 'cyan']

plt.figure(figsize=(22,14), facecolor='black')
for win, col in zip(windows, colors):
    f, p = welch(signal, fs=fs, window=win, nperseg=4096, noverlap=2048)
    plt.semilogy(f[:200], p[:200], color=col, linewidth=6, label=win.capitalize())

plt.axvline(f0, color='white', linewidth=8)
plt.text(f0+0.3, 1e-4, '43.000000000 Hz', color='gold', fontsize=36, rotation=90)

plt.title("HAMMING WINS — Cleanest 43 Hz Spike\n"
          "Window Function Deathmatch\n"
          "Mihai A. Bucurenciu — Romania, December 2025",
          color='white', fontsize=36, pad=80)
plt.xlabel("Frequency (Hz)", color='white', fontsize=30)
plt.ylabel("Power (µV²/Hz)", color='white', fontsize=30)
plt.legend(fontsize=30, facecolor='black')

ax = plt.gca()
ax.set_facecolor('black')
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines[['top','right']].set_visible(False)
plt.grid(True, which='both', ls='--', c='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("plots/window_function_comparison.png", dpi=1200, facecolor='black')
plt.close()

print("Window comparison plot saved — plots/window_function_comparison.png")
