#!/usr/bin/env python3
import mne
import numpy as np
import matplotlib.pyplot as plt

# Load subject_31 .edf
edf_path = 'data/raw_edf/subject_31_meditation_3h.edf'  # your file
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

# Higuchi FD
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

# DFA
def dfa(ts, order=1):
    ts = np.asarray(ts).flatten()
    y = np.cumsum(ts - ts.mean())
    scales = np.logspace(np.log10(4), np.log10(len(ts)//4), 15, dtype=int)
    F_n = []
    for n in scales:
        segments = len(ts) // n
        rms = []
        for v in range(segments):
            seg = y[v*n:(v+1)*n]
            x = np.arange(n)
            coeffs = np.polyfit(x, seg, order)
            trend = np.polyval(coeffs, x)
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        F_n.append(np.mean(rms))
    coeffs = np.polyfit(np.log(scales), np.log(F_n), 1)
    return coeffs[0]

hfd = higuchi_fd(data[0])
alpha = dfa(data[0])
print(f"Higuchi FD: {hfd:.4f}")
print(f"DFA α: {alpha:.4f}")

# Plot
plt.figure(figsize=(12,6))
plt.plot(times, data[0], 'gold', lw=2)
plt.title(f"Subject 31 (Meditation) — Full EEG\nHiguchi FD = {hfd:.3f}, DFA α = {alpha:.3f}")
plt.xlabel("Time [s]"); plt.ylabel("Amplitude [µV]")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("subject_31_higuchi_dfa.png", dpi=400)
plt.show()
