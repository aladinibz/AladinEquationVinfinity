"""
Subject 31 Meditation - Multifractal Detrended Fluctuation Analysis (MF-DFA)
Natural 3-hour sustained meditation
Proof of slow ego turbulence collapse to 43 Hz condensate order
ALADIN ∞ ℂ(t) — The Final Law
December 31, 2025
"""

import urllib.request
import mne
import numpy as np
import matplotlib.pyplot as plt
from MFDFA import MFDFA
from scipy.stats import linregress

# Download raw EDF
url = "https://raw.githubusercontent.com/aladinibz/AladinEquationVinfinity/main/data/raw_edf/subject_31_meditation_3h.edf"
filename = "subject_31_meditation_3h.edf"
urllib.request.urlretrieve(url, filename)

# Load and prepare signal
raw = mne.io.read_raw_edf(filename, preload=True)
if 'Cz' in raw.ch_names:
    raw.pick(['Cz'])
else:
    raw.pick_types(eeg=True)
raw.resample(128)
data = raw.get_data()
signal = np.mean(data, axis=0) if data.ndim > 1 else data.flatten()
signal = (signal - np.mean(signal)) / np.std(signal)

# MF-DFA parameters (balanced for accuracy/speed)
lag = np.unique(np.logspace(2, np.log10(len(signal)//20), 20).astype(int))
q = np.linspace(-4, 4, 9)
q = q[q != 0]

# Run MF-DFA
lag, dfa = MFDFA(signal, lag=lag, q=q, order=1)
min_len = min(len(lag), dfa.shape[1])
lag = lag[:min_len]
dfa = dfa[:, :min_len]

# Generalized Hurst
hq = []
for i in range(len(q)):
    valid = np.where(dfa[i] > 0)
    if len(valid[0]) > 3:
        hq.append(linregress(np.log(lag[valid]), np.log(dfa[i][valid])).slope)
    else:
        hq.append(0.5)

# Singularity spectrum
tau = q * np.array(hq) - 1
alpha = np.gradient(tau, q)
f_alpha = q * alpha - tau

# Save plots
plt.figure(figsize=(12,8))
for i in range(0, len(q), 2):
    plt.loglog(lag, dfa[i])
plt.title('Subject 31 Meditation - Fluctuation Functions')
plt.xlabel('Scale')
plt.ylabel('F(q,s)')
plt.grid(True)
plt.savefig('plots/subject_31_meditation_fluctuation_functions.png')
plt.close()

plt.figure(figsize=(10,6))
plt.plot(q, hq, 'o-', color='gold', linewidth=4)
plt.title('Subject 31 Meditation - Generalized Hurst h(q)')
plt.xlabel('q')
plt.ylabel('h(q)')
plt.grid()
plt.savefig('plots/subject_31_meditation_hq_curve.png')
plt.close()

plt.figure(figsize=(10,7))
plt.plot(alpha, f_alpha, 'ro-', linewidth=4)
plt.title('Subject 31 Meditation - Singularity Spectrum')
plt.xlabel('α')
plt.ylabel('f(α)')
plt.grid()
plt.savefig('plots/subject_31_meditation_singularity_spectrum.png')
plt.close()

# Δα timecourse
window_min = 15
win_samples = window_min * 60 * 128
n_win = len(signal) // win_samples
deltas = []
for i in range(n_win):
    win = signal[i*win_samples:(i+1)*win_samples]
    w_lag = np.unique(np.logspace(2, np.log10(len(win)//20), 15).astype(int))
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
            w_hq.append(0.5)
    w_tau = q * np.array(w_hq) - 1
    w_alpha = np.gradient(w_tau, q)
    deltas.append(w_alpha.max() - w_alpha.min())

time_min = np.arange(len(deltas)) * window_min + window_min/2
plt.figure(figsize=(14,7))
plt.plot(time_min, deltas, 'go-', linewidth=4)
plt.title('Subject 31 Meditation - Δα Collapse Over Time')
plt.xlabel('Time (minutes)')
plt.ylabel('Δα')
plt.grid()
plt.savefig('plots/subject_31_meditation_delta_alpha_timecourse.png')
plt.close()
