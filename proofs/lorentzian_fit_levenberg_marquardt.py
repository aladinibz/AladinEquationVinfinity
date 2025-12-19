#!/usr/bin/env python3
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

os.makedirs("plots", exist_ok=True)

# Load example .edf (change to your subject)
raw = mne.io.read_raw_edf('subject_17_5meo_18mg.edf', preload=True)
raw.filter(1, 100); raw.notch_filter(np.arange(50, 251, 50))
picks = ['AFz'] if 'AFz' in raw.ch_names else mne.pick_types(raw.info, eeg=True)[:1]

# Compute PSD
psd = raw.compute_psd(fmin=30, fmax=60, picks=picks)
freqs = psd.freqs
power = psd.get_data()[0]

# Lorentzian function
def lorentz(f, f0, gamma, A, offset):
    return A * (gamma/2)**2 / ((f - f0)**2 + (gamma/2)**2) + offset

# Initial guess + bounds
p0 = [43, 0.001, np.max(power), np.min(power)]
bounds = ([40, 1e-12, 0, 0], [46, 1, np.inf, np.inf])

# Fit with 'trf' (handles bounds)
popt, pcov = curve_fit(lorentz, freqs, power, p0=p0, bounds=bounds, method='trf')

f0, gamma, A, offset = popt
print(f"Peak frequency: {f0:.9f} Hz")
print(f"Linewidth Δf (FWHM): {gamma:.12f} Hz")
print(f"Q = f / Δf ≈ {f0 / gamma:.2e}")

# Plot
plt.figure(figsize=(12,8),dpi=1200)
plt.semilogy(freqs, power, 'gold', lw=3, label='Measured PSD')
plt.semilogy(freqs, lorentz(freqs, *popt), 'purple', lw=3, ls='--', label='Lorentzian Fit')
plt.axvline(f0, color='darkred', ls=':', lw=2)
plt.title(f"Lorentzian Fit\nΔf = {gamma:.2e} Hz, Q ≈ {f0/gamma:.2e}")
plt.xlabel("Frequency [Hz]"); plt.ylabel("Power")
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/lorentzian_linewidth_fit.png",dpi=1200)
plt.show()
