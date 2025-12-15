#!/usr/bin/env python3
import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Load subject_31 .edf
edf_path = 'data/raw_edf/subject_31_meditation_3h.edf'
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Preprocess
raw.filter(1, 100)
raw.notch_filter(np.arange(50, 251, 50))

# Pick AFz or fallback
if 'AFz' in raw.ch_names:
    picks = ['AFz']
else:
    picks = mne.pick_types(raw.info, eeg=True)[:1]

# Compute PSD (new MNE way — no import error)
psd = raw.compute_psd(fmin=1, fmax=100, picks=picks)
freqs = psd.freqs
power = psd.get_data()[0]  # first channel

# Plot spectrum
plt.figure(figsize=(12,8),dpi=400)
plt.semilogy(freqs, power, 'gold', lw=3)
plt.axvline(43, color='purple', ls='--', lw=4, label='43 Hz Lock')
plt.title("ALADIN ∞ ℂ(t) — Subject 31 (3h Meditation)\nFull Power Spectrum Shift",fontsize=18)
plt.xlabel("Frequency [Hz]",fontsize=14); plt.ylabel("Power (log)",fontsize=14)
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/subject_31_meditation_spectrum.png",dpi=400)
print("Saved: plots/subject_31_meditation_spectrum.png")
