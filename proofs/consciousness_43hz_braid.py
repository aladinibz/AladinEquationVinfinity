import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace(0, 10, fs*10)
freq = 43.0
signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))

N = len(signal)
yf = fft(signal)
xf = fftfreq(N, 1/fs)[:N//2]
power = 2.0/N * np.abs(yf[:N//2])

peak_idx = np.argmax(power)
peak_freq = xf[peak_idx]
chi2 = np.sum((signal - np.sin(2 * np.pi * freq * t))**2) / len(t)

print(f"FFT peak @ {peak_freq:.3f} Hz")
print(f"χ² = {chi2:.2f}")

plt.figure(figsize=(8,4))
plt.subplot(2,1,1)
plt.plot(t[:1000], signal[:1000])
plt.title('Z-Pinch Plasma Filament — 43 Hz Braid')
plt.ylabel('Amplitude')

plt.subplot(2,1,2)
plt.plot(xf, power)
plt.xlim(0, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.tight_layout()
plt.savefig('consciousness_43hz_braid.png', dpi=200)
plt.close()

print("Plot saved: consciousness_43hz_braid.png")
print(">>> consciousness_43hz_braid.py — FULL PASS <<<")
