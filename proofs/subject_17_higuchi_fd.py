#!/usr/bin/env python3
import mne
import numpy as np
import matplotlib.pyplot as plt

# Load the .edf file
edf_path = 'data/raw_edf/subject_17_5meo_18mg.edf'  # path in repo
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

# Higuchi FD function
def higuchi_fd(ts, k_max=20):
    ts = np.asarray(ts).flatten()
    N = len(ts)
    L = []
    for k in range(1, k_max+1):
        Lk = []
        for m in range(k):
            num = (N - m) // k
            if num < 2: continue
            Lmk = np.sum(np.abs(np.diff(ts[m::k]))) * (N-1) / (num * k)
            Lk.append(Lmk)
        L.append(np.mean(Lk))
    log_L = np.log(L)
    log_k = np.log(np.arange(1, k_max+1)[:len(L)])
    return -np.polyfit(log_k, log_L, 1)[0]

hfd = higuchi_fd(data[0])
print(f"Higuchi FD on full subject_17 EEG: {hfd:.4f}")

# Plot and save
plt.figure(figsize=(12,6))
plt.plot(times, data[0], 'gold', lw=2)
plt.title(f"Subject 17 (5-MeO-DMT) — Full EEG\nHiguchi FD = {hfd:.3f}")
plt.xlabel("Time [s]"); plt.ylabel("Amplitude [µV]")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("subject_17_higuchi_fd.png", dpi=400)
