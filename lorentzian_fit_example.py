#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Simulated PSD data (replace with your real MNE PSD)
freqs = np.linspace(30, 60, 1000)
power = 1e-10 * (1 / ((freqs - 43)**2 + 1e-6) + 0.1)  # Lorentzian + noise

# Lorentzian model
def lorentz(f, f0, gamma, A, offset):
    return A * (gamma/2)**2 / ((f - f0)**2 + (gamma/2)**2) + offset

# Initial guess + bounds
p0 = [43, 0.001, np.max(power), np.min(power)]
bounds = ([40, 1e-12, 0, 0], [46, 1, np.inf, np.inf])

# Fit with 'trf' (handles bounds)
popt, pcov = curve_fit(lorentz, freqs, power, p0=p0, bounds=bounds, method='trf')

f0, gamma, A, offset = popt
print(f"Peak: {f0:.9f} Hz")
print(f"Δf (FWHM): {gamma:.12f} Hz")
print(f"Q = f / Δf ≈ {f0 / gamma:.2e}")

# Plot
plt.figure(figsize=(12,8),dpi=400)
plt.semilogy(freqs, power, 'gold', lw=3, label='Data')
plt.semilogy(freqs, lorentz(freqs, *popt), 'purple', lw=3, ls='--', label='Fit')
plt.axvline(f0, color='darkred', ls=':', lw=2)
plt.title(f"Lorentzian Fit\nΔf = {gamma:.2e} Hz, Q ≈ {f0/gamma:.2e}")
plt.xlabel("Frequency [Hz]"); plt.ylabel("Power")
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/lorentzian_fit_example.png",dpi=400)
plt.show()
