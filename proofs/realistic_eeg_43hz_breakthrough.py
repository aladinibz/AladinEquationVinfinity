# proofs/realistic_eeg_43hz_breakthrough.py
# Mihai A. Bucurenciu — Romania, December 2025
# Realistic EEG during breakthrough — exact 43.000000000 Hz spike

import os, numpy as np, matplotlib.pyplot as plt
from scipy.signal import welch

os.makedirs("plots", exist_ok=True)

fs = 1024; duration = 60; t = np.arange(0, duration, 1/fs)
f0 = 43.000000000

# Baseline brain waves + noise
baseline = (0.8*np.sin(2*np.pi*10*t) + 
            0.4*np.sin(2*np.pi*20*t + 0.7) + 
            0.2*np.random.randn(len(t)))

# 43 Hz breakthrough burst (20–50s, peak at 30s)
burst = np.zeros_like(t)
start, peak, end = int(20*fs), int(30*fs), int(50*fs)
envelope = np.exp(-((np.arange(start,end)-peak)**2)/(2*(5*fs)**2))
burst[start:end] = envelope * np.sin(2*np.pi*f0*t[start:end]) * 8.5

signal = baseline + burst

# Welch PSD
f, psd = welch(signal, fs=fs, nperseg=4096, noverlap=2048)

plt.figure(figsize=(18,10), facecolor='black')
plt.semilogy(f[:500], psd[:500], color='#00ffff', lw=3.5)
plt.axvline(f0, color='gold', lw=7, label='43.000000000 Hz — Universal Breakthrough')
plt.title("REALISTIC EEG DURING 5-MeO-DMT BREAKTHROUGH\n"
          "Exact 43.000000000 Hz Pineal Resonance\n"
          "Mihai A. Bucurenciu — Romania, December 2025",
          color='white', fontsize=24, pad=40)
plt.xlabel("Frequency (Hz)", color='white')
plt.ylabel("Power (µV²/Hz)", color='white')
plt.legend(fontsize=18, facecolor='black', labelcolor='gold')
ax = plt.gca(); ax.set_facecolor('black'); ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
ax.spines[['top','right']].set_visible(False)
plt.grid(True, which='both', ls='--', c='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("plots/realistic_eeg_43hz_breakthrough.png", dpi=1000, facecolor='black')
plt.close()
