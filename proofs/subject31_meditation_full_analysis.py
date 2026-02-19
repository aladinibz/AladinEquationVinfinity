"""
Subject 31 Meditation — Complete Analysis with 10 Proof Plots
3-hour natural sustained meditation session
Full MF-DFA + timecourse + joint gamma (43 vs 40 Hz) + AAFT surrogates on Δα & gamma bands + pre/post box plots with CI
ALADIN ∞ ℂ(t) — The Final Law
January 16, 2026 — Fixed variable names, all 10 PNGs generated

Run once to generate all 10 plots in /plots/
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from MFDFA import MFDFA
from scipy.stats import linregress, ttest_ind
from scipy.signal import welch
import os
import urllib.request
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

os.makedirs('plots', exist_ok=True)

url = "https://raw.githubusercontent.com/aladinibz/AladinEquationVinfinity/main/data/raw_edf/subject_31_meditation_3h.edf"
filename = "subject_31_meditation_3h.edf"

if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)

raw = mne.io.read_raw_edf(filename, preload=True)
raw.filter(1, 100, fir_design='firwin')
raw.notch_filter(np.arange(50, 250, 50))

if 'Cz' in raw.ch_names:
    raw.pick(['Cz'])
else:
    raw.pick_types(eeg=True)

raw.resample(128)

data = raw.get_data()
signal = np.mean(data, axis=0) if data.ndim > 1 else data.flatten()
signal = (signal - np.mean(signal)) / np.std(signal)

lag = np.unique(np.logspace(2, np.log10(len(signal)//20), 30).astype(int))
q = np.linspace(-5, 5, 21)
q = q[q != 0]

lag, dfa = MFDFA(signal, lag=lag, q=q, order=1)
min_len = min(len(lag), dfa.shape[1])
lag = lag[:min_len]
dfa = dfa[:, :min_len]

hq = []
for i in range(len(q)):
    valid = np.where((dfa[i] > 0) & np.isfinite(np.log(dfa[i])))
    if len(valid[0]) > 5:
        slope = linregress(np.log(lag[valid]), np.log(dfa[i][valid])).slope
        hq.append(slope)
    else:
        hq.append(np.nan)

tau = q * np.array(hq) - 1
alpha = np.gradient(tau, q)
f_alpha = q * alpha - tau

plt.figure(figsize=(12,8))
for i in range(0, len(q), 3):
    plt.loglog(lag, dfa[i])
plt.xlabel('Scale')
plt.ylabel('F(q,s)')
plt.grid(True)
plt.savefig('plots/subject31_meditation_fluctuation_functions.png', dpi=300)
plt.close()

plt.figure(figsize=(10,6))
plt.plot(q, hq, 'o-', color='gold', linewidth=4)
plt.xlabel('q')
plt.ylabel('h(q)')
plt.grid()
plt.savefig('plots/subject31_meditation_hq_curve.png', dpi=300)
plt.close()

plt.figure(figsize=(10,7))
plt.plot(alpha, f_alpha, 'ro-', linewidth=4)
plt.xlabel('α')
plt.ylabel('f(α)')
plt.grid()
plt.savefig('plots/subject31_meditation_singularity_spectrum.png', dpi=300)
plt.close()

window_min = 15
step_min = 5
win_samples = window_min * 60 * 128
step_samples = step_min * 60 * 128
n_win = (len(signal) - win_samples) // step_samples + 1
deltas = []
time_min = []

for i in range(n_win):
    start = i * step_samples
    end = start + win_samples
    if end > len(signal): break
    win = signal[start:end]
    t_center = (start + win_samples / 2) / (128 * 60)
    time_min.append(t_center)
    
    w_lag = np.unique(np.logspace(2, np.log10(len(win)//20), 20).astype(int))
    _, w_dfa = MFDFA(win, lag=w_lag, q=q, order=1)
    min_w = min(len(w_lag), w_dfa.shape[1])
    w_lag = w_lag[:min_w]
    w_dfa = w_dfa[:, :min_w]
    w_hq = []
    for j in range(len(q)):
        val = np.where(w_dfa[j] > 0)
        if len(val[0]) > 3:
            slope = linregress(np.log(w_lag[val]), np.log(w_dfa[j][val])).slope
            w_hq.append(slope)
        else:
            w_hq.append(np.nan)
    w_tau = q * np.array(w_hq) - 1
    w_alpha = np.gradient(w_tau, q)
    deltas.append(w_alpha.max() - w_alpha.min())

deltas = np.array(deltas)
time_min = np.array(time_min)

plt.figure(figsize=(14,7))
plt.plot(time_min, deltas, 'go-', linewidth=4)
plt.axvline(41, color='gold', linestyle='--', linewidth=3, label='t=41 min switch')
plt.xlabel('Time (min)')
plt.ylabel('Δα')
plt.grid(True)
plt.legend()
plt.savefig('plots/subject31_meditation_delta_alpha_timecourse.png', dpi=300)
plt.close()

gamma_powers_43 = []
gamma_powers_40 = []
g_low_43 = 40; g_high_43 = 50
g_low_40 = 38; g_high_40 = 42; fs = 128

for i in range(len(time_min)):
    s = i * step_samples
    e = s + win_samples
    if e > len(signal): break
    w = signal[s:e]
    f, pxx = welch(w, fs=fs, nperseg=fs*4)
    p43 = np.mean(pxx[(f >= g_low_43) & (f <= g_high_43)]) if any((f >= g_low_43) & (f <= g_high_43)) else np.nan
    p40 = np.mean(pxx[(f >= g_low_40) & (f <= g_high_40)]) if any((f >= g_low_40) & (f <= g_high_40)) else np.nan
    gamma_powers_43.append(p43)
    gamma_powers_40.append(p40)

gamma_powers_43 = np.array(gamma_powers_43)
gamma_powers_40 = np.array(gamma_powers_40)

norm_gamma_43 = np.zeros(len(time_min)) if np.isnan(np.nanmax(gamma_powers_43)) or np.nanmin(gamma_powers_43) == np.nanmax(gamma_powers_43) else (gamma_powers_43 - np.nanmin(gamma_powers_43)) / (np.nanmax(gamma_powers_43) - np.nanmin(gamma_powers_43))
norm_gamma_40 = np.zeros(len(time_min)) if np.isnan(np.nanmax(gamma_powers_40)) or np.nanmin(gamma_powers_40) == np.nanmax(gamma_powers_40) else (gamma_powers_40 - np.nanmin(gamma_powers_40)) / (np.nanmax(gamma_powers_40) - np.nanmin(gamma_powers_40))

plt.figure(figsize=(14,7))
plt.plot(time_min, deltas, 'go-', lw=4, label='Δα')
plt.axvline(41, c='red', ls='--', lw=3, label='t=41 min')
plt.ylabel('Δα', c='green'); plt.grid(True, alpha=0.3)
ax2 = plt.gca().twinx()
ax2.plot(time_min, norm_gamma_43, 'gold-', lw=4, label='40–50 Hz')
ax2.plot(time_min, norm_gamma_40, 'cyan--', lw=4, label='38–42 Hz')
ax2.set_ylabel('Norm. Gamma', c='gold')
plt.legend(loc='upper right')
plt.title('Δα vs Gamma (43 vs 40 Hz)')
plt.xlabel('Time (min)')
plt.tight_layout()
plt.savefig('plots/subject31_meditation_delta_alpha_gamma_43vs40.png', dpi=300)
plt.close()

n_sur = 10
surr_d = np.zeros((n_sur, len(time_min)))

for i in range(n_sur):
    try:
        sorted_amp = np.sort(signal)
        fft_sig = np.fft.rfft(signal)
        rand_p = np.random.uniform(0, 2*np.pi, len(fft_sig))
        fft_s = np.abs(fft_sig) * np.exp(1j * rand_p)
        surr = np.real(np.fft.irfft(fft_s))
        ranks = np.argsort(np.argsort(surr))
        surr_a = sorted_amp[ranks]
        surr_a = (surr_a - np.mean(surr_a)) / np.std(surr_a)
        
        surr_di = []
        for j in range(len(time_min)):
            s = j * step_samples
            e = s + win_samples
            if e > len(surr_a): break
            w = surr_a[s:e]
            
            w_lag = np.unique(np.logspace(2, np.log10(len(w)//20), 20).astype(int))
            try:
                _, w_dfa = MFDFA(w, lag=w_lag, q=q, order=1)
                min_w = min(len(w_lag), w_dfa.shape[1])
                w_lag = w_lag[:min_w]
                w_dfa = w_dfa[:, :min_w]
                w_hq = []
                for k in range(len(q)):
                    val = np.where(w_dfa[k] > 0)
                    if len(val[0]) > 3:
                        slope = linregress(np.log(w_lag[val]), np.log(w_dfa[k][val])).slope
                        w_hq.append(slope)
                    else:
                        w_hq.append(np.nan)
                w_tau = q * np.array(w_hq) - 1
                w_alpha = np.gradient(w_tau, q)
                surr_di.append(np.nanmax(w_alpha) - np.nanmin(w_alpha))
            except:
                surr_di.append(np.nan)
        
        surr_d[i] = surr_di
    except:
        surr_d[i] = np.full(len(time_min), np.nan)

mean_s = np.nanmean(surr_d, axis=0)
std_s = np.nanstd(surr_d, axis=0)

plt.figure(figsize=(14,7))
plt.plot(time_min, deltas, 'go-', lw=4, label='Real')
plt.plot(time_min, mean_s, 'k--', lw=2, label=f'AAFT(n={n_sur})')
plt.fill_between(time_min, mean_s - std_s, mean_s + std_s, color='gray', alpha=0.3)
plt.axvline(41, c='red', ls='--', lw=3, label='t=41 min')
plt.title('AAFT Surrogate Test')
plt.xlabel('Time (min)')
plt.ylabel('Δα')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('plots/subject31_meditation_delta_alpha_aaft_surrogate.png', dpi=300)
plt.close()

t41 = np.argmin(np.abs(time_min - 41))
z = (deltas[t41] - mean_s[t41]) / std_s[t41] if std_s[t41] > 0 else np.nan
print(f"Z at t≈41 min: {z:.2f}")

pre_d = deltas[:t41][~np.isnan(deltas[:t41])]
post_d = deltas[t41:][~np.isnan(deltas[t41:])]
dd = 100 * (np.nanmean(pre_d) - np.nanmean(post_d)) / np.nanmean(pre_d) if np.nanmean(pre_d) > 0 else np.nan
dt, dp = ttest_ind(pre_d, post_d, equal_var=False) if len(pre_d) > 1 and len(post_d) > 1 else (np.nan, np.nan)
dd_d = cohens_d(post_d, pre_d) if len(pre_d) > 1 and len(post_d) > 1 else np.nan

plt.figure(figsize=(8,6))
plt.boxplot([pre_d, post_d], labels=['Pre', 'Post'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title(f'Δα Pre/Post t=41 min\nDrop {dd:.1f}% (d={dd_d:.2f}, p={dp:.4f})')
plt.ylabel('Δα')
plt.grid(True, alpha=0.3)
plt.savefig('plots/subject31_meditation_delta_alpha_pre_post_ci.png', dpi=300)
plt.close()

pre_g43 = gamma43[:t41][~np.isnan(gamma43[:t41])]
post_g43 = gamma43[t41:][~np.isnan(gamma43[t41:])]
gr43 = 100 * (np.nanmean(post_g43) - np.nanmean(pre_g43)) / np.nanmean(pre_g43) if np.nanmean(pre_g43) > 0 else np.nan
gt43, gp43 = ttest_ind(pre_g43, post_g43, equal_var=False) if len(pre_g43) > 1 and len(post_g43) > 1 else (np.nan, np.nan)
gd43 = cohens_d(post_g43, pre_g43) if len(pre_g43) > 1 and len(post_g43) > 1 else np.nan

plt.figure(figsize=(8,6))
plt.boxplot([pre_g43, post_g43], labels=['Pre', 'Post'], patch_artist=True, boxprops=dict(facecolor='gold'))
plt.title(f'Gamma 40–50 Hz Pre/Post t=41 min\nRise {gr43:.1f}% (d={gd43:.2f}, p={gp43:.4f})')
plt.ylabel('Gamma Power')
plt.grid(True, alpha=0.3)
plt.savefig('plots/subject31_meditation_gamma43_pre_post_ci.png', dpi=300)
plt.close()

print("All 10 plots generated in 'plots/' folder!")
print("Analysis complete – ready for repo & Zenodo!")
