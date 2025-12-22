#!/usr/bin/env python3
import mne
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

raw = mne.io.read_raw_edf('subject_31_meditation_3h.edf', preload=True)
raw.filter(1, 100); raw.notch_filter(np.arange(50, 251, 50))

picks = ['AFz'] if 'AFz' in raw.ch_names else mne.pick_types(raw.info, eeg=True)[:1]

n_fft = 2048
psd_full = raw.compute_psd(fmin=1, fmax=100, picks=picks, n_fft=n_fft)
freqs = psd_full.freqs
power_full = psd_full.get_data()[0]

data_rest = raw.get_data(picks, start=0, stop=30000)[0].flatten()
data_deep = raw.get_data(picks, start=len(raw.times)-30000, stop=len(raw.times))[0].flatten()

psd_rest = mne.time_frequency.psd_array_welch(data_rest, raw.info['sfreq'], fmin=1, fmax=100, n_fft=n_fft)
psd_deep = mne.time_frequency.psd_array_welch(data_deep, raw.info['sfreq'], fmin=1, fmax=100, n_fft=n_fft)

power_rest = psd_rest[0]
power_deep = psd_deep[0]

coh_rest = np.max(power_rest[np.abs(freqs - 43) < 1])
coh_deep = np.max(power_deep[np.abs(freqs - 43) < 1])

plt.figure(figsize=(12,8),dpi=1200)
plt.semilogy(freqs, power_full, 'gray', lw=2, alpha=0.7, label='Full')
plt.semilogy(freqs, power_rest, 'blue', lw=4, label=f'Rest ({coh_rest:.2e})')
plt.semilogy(freqs, power_deep, 'gold', lw=4, label=f'Deep ({coh_deep:.2e})')
plt.axvline(43, color='purple', ls='--', lw=4, label='43 Hz')
plt.title("Subject 31 — Rest vs Deep PSD with Coherence Metrics",fontsize=18)
plt.xlabel("Freq [Hz]"); plt.ylabel("Power (log)")
plt.legend(fontsize=12); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/subject_31_psd_coherence_metrics.png",dpi=1200)
print("Saved: plots/subject_31_psd_coherence_metrics.png")
plt.show()
