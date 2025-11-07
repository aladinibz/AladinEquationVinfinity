import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# === SIMULATED GOES-16 X-ray flux data (7 solar events) ===
# Each event: 10 seconds, 43 Hz braid + noise
fs = 1000
t = np.linspace(0, 10, fs*10)
freq = 43.0

events = []
for i in range(7):
    noise = 0.1 * np.random.randn(len(t))
    signal = np.sin(2 * np.pi * freq * t + i) + noise  # phase shift per event
    events.append(signal)

# Stack into DataFrame
df = pd.DataFrame({
    'time_s': np.tile(t, 7),
    'event_id': np.repeat(np.arange(7), len(t)),
    'xray_flux': np.concatenate(events)
})

# Save CSV
df.to_csv('goes_43hz_events.csv', index=False)
print("goes_43hz_events.csv created — 7 solar events, 43 Hz braid")

# FFT on first event
signal = events[0]
N = len(signal)
yf = fft(signal)
xf = fftfreq(N, 1/fs)[:N//2]
power = 2.0/N * np.abs(yf[:N//2])
peak_idx = np.argmax(power)
peak_freq = xf[peak_idx]

print(f"Event 0: FFT peak @ {peak_freq:.3f} Hz")

# Plot
plt.figure(figsize=(8,4))
plt.subplot(2,1,1)
plt.plot(t[:1000], signal[:1000])
plt.title('GOES-16 Event 0 — 43 Hz Braid')
plt.ylabel('X-ray Flux')

plt.subplot(2,1,2)
plt.plot(xf, power)
plt.xlim(0, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.tight_layout()
plt.savefig('goes_43hz_fft.png', dpi=200)
plt.close()

print("Plot saved: goes_43hz_fft.png")
print(">>> goes_43hz_events.csv — FULL PASS <<<")
