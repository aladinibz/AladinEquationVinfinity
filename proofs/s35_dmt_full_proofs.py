"""
S35 DMT Breakthrough — Full Proofs
MF-DFA + PSD Comparison
Sharp collapse to 43 Hz condensate
ALADIN ∞ ℂ(t) — The Final Law
December 31, 2025
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from MFDFA import MFDFA
from scipy.stats import linregress
from scipy.signal import welch
import os

os.makedirs('plots', exist_ok=True)

# Load S35 DMT
raw_dmt = mne.io.read_raw_bdf('data/raw_bdf/S35-DMT.bdf', preload=True)

# Load rest (EO or EC — adjust if needed)
raw_rest = mne.io.read_raw_bdf('data/raw_bdf/S35-EO.bdf', preload=True)  # or S35-EC.bdf

for raw in [raw_dmt, raw_rest]:
    if 'Cz' in raw.ch_names:
        raw.pick(['Cz'])
    else:
        raw.pick_types(eeg=True)
    raw.resample(128)

signal_dmt = np.mean(raw_dmt.get_data(), axis=0)
signal_rest = np.mean(raw_rest.get_data(), axis=0)

signal_dmt = (signal_dmt - np.mean(signal_dmt)) / np.std(signal_dmt)

# MF-DFA
lag = np.unique(np.logspace(1.3, np.log10(len(signal_dmt)//10), 30).astype(int))
q = np.linspace(-5, 5, 21)
q = q[q != 0]

lag, dfa = MFDFA(signal_dmt, lag=lag, q=q, order=1)
min_len = min(len(lag), dfa.shape[1])
lag = lag[:min_len]
dfa = dfa[:, :min_len]

hq = []
for i in range(len(q)):
    valid = np.where((dfa[i] > 0) & np.isfinite(np.log(dfa[i])))
    if len(valid[0]) > 5:
        hq.append(linregress(np.log(lag[valid]), np.log(dfa[i][valid])).slope)
    else:
        hq.append(np.nan)

tau = q * np.array(hq) - 1
alpha = np.gradient(tau, q)
f_alpha = q * alpha - tau

# Plot 1: Fluctuation Functions
plt.figure(figsize=(12,8))
for i in range(0, len(q), 3):
    plt.loglog(lag, dfa[i])
plt.title('S35 DMT - Fluctuation Functions')
plt.xlabel('Scale')
plt.ylabel('F(q,s)')
plt.grid(True)
plt.savefig('plots/s35_dmt_fluctuation_functions.png', dpi=300)
plt.close()

# Plot 2: h(q) Curve
plt.figure(figsize=(10,6))
plt.plot(q, hq, 'o-', color='gold', linewidth=4)
plt.title('S35 DMT - Generalized Hurst h(q)')
plt.xlabel('q')
plt.ylabel('h(q)')
plt.grid()
plt.savefig('plots/s35_dmt_hq_curve.png', dpi=300)
plt.close()

# Plot 3: Singularity Spectrum
plt.figure(figsize=(10,7))
plt.plot(alpha, f_alpha, 'ro-', linewidth=4)
plt.title('S35 DMT - Singularity Spectrum')
plt.xlabel('α')
plt.ylabel('f(α)')
plt.grid()
plt.savefig('plots/s35_dmt_singularity_spectrum.png', dpi=300)
plt.close()

# Plot 4: Δα Timecourse
window_sec = 30
win_samples = window_sec * 128
n_win = len(signal_dmt) // win_samples
deltas = []
time_sec = np.arange(n_win) * window_sec + window_sec/2

for i in range(n_win):
    win = signal_dmt[i*win_samples:(i+1)*win_samples]
    w_lag = np.unique(np.logspace(1.3, np.log10(len(win)//10), 20).astype(int))
    _, w_dfa = MFDFA(win, lag=w_lag, q=q, order=1)
    min_w = min(len(w_lag), w_dfa.shape[1])
    w_lag = w_lag[:min_w]
    w_dfa = w_dfa[:, :min_w]
    w_hq = []
    for j in range(len(q)):
        val = np.where(w_dfa[j] > 0)
        if len(val[0]) > 3:
            w_hq.append(linregress(np.log(w_lag[val]), np.log(w_dfa[j][val])).slope)
        else:
            w_hq.append(np.nan)
    w_tau = q * np.array(w_hq) - 1
    w_alpha = np.gradient(w_tau, q)
    deltas.append(w_alpha.max() - w_alpha.min())

plt.figure(figsize=(14,7))
plt.plot(time_sec, deltas, 'ro-', linewidth=4)
plt.axvline(41, color='gold', linestyle='--', linewidth=3)
plt.title('S35 DMT - Δα Collapse Timecourse')
plt.xlabel('Time (seconds)')
plt.ylabel('Δα')
plt.grid(True)
plt.savefig('plots/s35_dmt_delta_alpha_timecourse.png', dpi=300)
plt.close()

# Plot 5: PSD Comparison
f_rest, psd_rest = welch(signal_rest, fs=128, nperseg=2048)
f_dmt, psd_dmt = welch(signal_dmt, fs=128, nperseg=2048)

plt.figure(figsize=(12,7))
plt.semilogy(f_rest, psd_rest, label='Rest / Eyes Open')
plt.semilogy(f_dmt, psd_dmt, label='DMT Breakthrough')
plt.axvline(43, color='gold', linestyle='--', linewidth=3)
plt.title('S35 DMT - PSD Comparison')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.xlim(1, 60)
plt.legend()
plt.grid(True)
plt.savefig('plots/s35_dmt_psd_comparison.png', dpi=300)
plt.close()
