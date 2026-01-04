"""
Wiener Filter Implementation for LArTPC Signal Deconvolution
Optimal linear filter — minimizes MSE
ALADIN ∞ ℂ(t) — The Final Law
January 04, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# Synthetic LArTPC signal
t = np.linspace(0, 1000, 1000)  # time samples
true_signal = np.zeros(1000)
true_signal[400:450] = 1.0  # step ionization

# Response function (electronics + field)
response = np.exp(- (t - 500)**2 / (2 * 50**2))  # Gaussian-like

# Convolved signal
convolved = np.convolve(true_signal, response, mode='same') / np.sum(response)

# Add noise
noise = 0.05 * np.random.randn(len(t))
measured = convolved + noise

# FFT
S = np.fft.rfft(measured)
R = np.fft.rfft(response)
freqs = np.fft.rfftfreq(len(t))

# Power spectra
P_s = np.abs(S)**2  # signal power (measured)
P_n = np.mean(np.abs(np.fft.rfft(noise))**2)  # noise power (constant approximation)

# Wiener filter
W = np.conj(R) / (np.abs(R)**2 + P_n / P_s)

# Deconvolved
Q_hat = W * S
deconvolved = np.fft.irfft(Q_hat)

# Plot
plt.figure(figsize=(14,10))
plt.subplot(3,1,1)
plt.plot(t, true_signal, label='True ionization', color='gold', linewidth=3)
plt.title('True Signal')
plt.grid(True, alpha=0.3)

plt.subplot(3,1,2)
plt.plot(t, measured, label='Measured (convolved + noise)', color='gray')
plt.title('Raw LArTPC Signal')
plt.grid(True, alpha=0.3)

plt.subplot(3,1,3)
plt.plot(t, deconvolved, label='Wiener deconvolved', color='purple', linewidth=3)
plt.plot(t, true_signal, '--', label='True (reference)', color='gold')
plt.title('Wiener Filter Deconvolution Result')
plt.xlabel('Time samples')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/wiener_filter_lartpc_deconvolution.png', dpi=400)
plt.close()

print("Wiener filter deconvolution saved — LArTPC signal recovered")
