#!/usr/bin/env python3
import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Load both .edf (upload in Colab first)
raw17 = mne.io.read_raw_edf('subject_17_5meo_18mg.edf', preload=True)
raw23 = mne.io.read_raw_edf('subject_23_dmt_35mg.edf', preload=True)

# Preprocess
raw17.filter(1, 100); raw23.filter(1, 100)
raw17.notch_filter(np.arange(50, 251, 50)); raw23.notch_filter(np.arange(50, 251, 50))

# Pick AFz or fallback
picks17 = ['AFz'] if 'AFz' in raw17.ch_names else mne.pick_types(raw17.info, eeg=True)[:1]
picks23 = ['AFz'] if 'AFz' in raw23.ch_names else mne.pick_types(raw23.info, eeg=True)[:1]

data17, times17 = raw17[picks17, :]
data23, times23 = raw23[picks23, :]

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

hfd17 = higuchi_fd(data17[0])
hfd23 = higuchi_fd(data23[0])
print(f"Higuchi FD subject 17 (5-MeO): {hfd17:.4f}")
print(f"Higuchi FD subject 23 (DMT): {hfd23:.4f}")

# Plot overlay
plt.figure(figsize=(12,6))
plt.plot(times17, data17[0], 'gold', lw=2, alpha=0.8, label=f"Subject 17 (5-MeO) FD={hfd17:.3f}")
plt.plot(times23, data23[0], 'darkorange', lw=2, alpha=0.8, label=f"Subject 23 (DMT) FD={hfd23:.3f}")
plt.title("ALADIN ∞ ℂ(t) — Subject 17 vs 23 Breakthrough EEG\nHiguchi FD Comparison")
plt.xlabel("Time [s]"); plt.ylabel("Amplitude [µV]")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("plots/subject_17_vs_23_higuchi.png", dpi=400)
print("Saved: plots/subject_17_vs_23_higuchi.png")
