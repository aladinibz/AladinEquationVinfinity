#!/usr/bin/env python3
import mne
import matplotlib.pyplot as plt

# Load real .edf
edf_path = 'data/raw_edf/subject_17_5meo_18mg.edf'
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Preprocess
raw.filter(1, 100)
raw.notch_filter(np.arange(50, 251, 50))

# Pick AFz or fallback
if 'AFz' in raw.ch_names:
    picks = ['AFz']
else:
    picks = mne.pick_types(raw.info, eeg=True)[:1]

data, times = raw[picks, :]

# Plot + save
plt.figure(figsize=(12,6))
plt.plot(times, data[0], 'gold', lw=2)
plt.title("Subject 17 (5-MeO-DMT) — Full Breakthrough EEG")
plt.xlabel("Time [s]"); plt.ylabel("Amplitude [µV]")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("subject_17_full_eeg.png", dpi=400)
plt.show()
