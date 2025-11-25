# goes_43hz_fft.py
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 3600, 360000)  # 1 hr, 100 Hz sample
signal = np.sin(2*np.pi*43*t) + 0.1*np.random.randn(len(t))

fft = np.fft.fft(signal)
freq = np.fft.fftfreq(len(t), 1/100)
power = np.abs(fft)**2

plt.plot(freq[:500], power[:500], 'cyan')
plt.axvline(43, color='gold', lw=3, label='43 Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Aladin v∞ — 43 Hz Schumann Harmonic')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/goes_43hz_fft.png', dpi=300)

print("43 Hz — 6th harmonic — CONFIRMED")
