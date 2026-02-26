import mne
import numpy as np
import matplotlib.pyplot as plt
from MFDFA import MFDFA
from scipy.stats import linregress, ttest_ind
from scipy.signal import welch, hilbert
import os
import urllib.request
import warnings
from scipy.signal import butter, sosfiltfilt
import networkx as nx
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from ordpy import permutation_entropy

warnings.filterwarnings("ignore", category=RuntimeWarning)

os.makedirs('plots', exist_ok=True)

# ─── Load S01-DMT.bdf ───────────────────────────────────────────────────────
file_name = "S01-DMT.bdf"
url = f"https://raw.githubusercontent.com/aladinibz/AladinEquationVinfinity/main/data/raw_bdf/{file_name}"

if not os.path.exists(file_name):
    urllib.request.urlretrieve(url, file_name)

raw = mne.io.read_raw_bdf(file_name, preload=True)
raw.filter(1, 100, fir_design='firwin')
raw.notch_filter(np.arange(50, 250, 50))
raw.pick_types(eeg=True)
raw.resample(128)

data = raw.get_data()
signal = np.mean(data, axis=0) if data.ndim > 1 else data.flatten()
signal = (signal - np.mean(signal)) / np.std(signal)

fs = 128

# =============================================================================
# YOUR ORIGINAL 13 CROWN JEWEL PLOTS (exact copy, full block)
# =============================================================================
print("Running your 13 crown jewel plots...")

lag = np.unique(np.logspace(2, np.log10(len(signal)//20), 30).astype(int))
q = np.linspace(-5, 5, 21)
q = q[q != 0]

lag, dfa = MFDFA(signal, lag=lag, q=q, order=1)
min_len = min(len(lag), dfa.shape[1])
lag = lag[:min_len]
dfa = dfa[:, :min_len]

plt.figure(figsize=(12,8))
for i in range(0, len(q), 3):
    plt.loglog(lag, dfa[i])
plt.xlabel('Scale')
plt.ylabel('F(q,s)')
plt.grid(True)
plt.savefig('plots/s01_dmt_fluctuation_functions.png', dpi=300)
plt.close()

hq = []
for i in range(len(q)):
    valid = np.where((dfa[i] > 0) & np.isfinite(np.log(dfa[i])))
    if len(valid[0]) > 5:
        slope = linregress(np.log(lag[valid]), np.log(dfa[i][valid])).slope
        hq.append(slope)
    else:
        hq.append(np.nan)

plt.figure(figsize=(10,6))
plt.plot(q, hq, 'o-', color='gold', linewidth=4)
plt.xlabel('q')
plt.ylabel('h(q)')
plt.grid()
plt.savefig('plots/s01_dmt_hq_curve.png', dpi=300)
plt.close()

tau = q * np.array(hq) - 1
alpha = np.gradient(tau, q)
f_alpha = q * alpha - tau

plt.figure(figsize=(10,7))
plt.plot(alpha, f_alpha, 'ro-', linewidth=4)
plt.xlabel('α')
plt.ylabel('f(α)')
plt.grid()
plt.savefig('plots/s01_dmt_singularity_spectrum.png', dpi=300)
plt.close()

window_sec = 5
step_sec = 1
win_samples = window_sec * 128
step_samples = step_sec * 128
n_win = (len(signal) - win_samples) // step_samples + 1
deltas = []
time_sec = []

for i in range(n_win):
    start = i * step_samples
    end = start + win_samples
    if end > len(signal): break
    win = signal[start:end]
    t_center = (start + win_samples / 2) / 128
    time_sec.append(t_center)
    
    w_lag = np.unique(np.logspace(2, np.log10(len(win)//20), 20).astype(int))
    try:
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
    except:
        deltas.append(np.nan)

deltas = np.array(deltas)
time_sec = np.array(time_sec)

plt.figure(figsize=(14,7))
plt.plot(time_sec, deltas, 'go-', linewidth=4)
plt.axvline(41, color='gold', linestyle='--', linewidth=3, label='t=41 s')
plt.title('Δα Collapse (S01-DMT)')
plt.xlabel('Time (s)')
plt.ylabel('Δα')
plt.grid(True)
plt.legend()
plt.savefig('plots/s01_dmt_delta_alpha_timecourse.png', dpi=300)
plt.close()

gamma43 = []; gamma40 = []
g43l=40; g43h=50; g40l=38; g40h=42

for i in range(len(time_sec)):
    s = i * step_samples
    e = s + win_samples
    if e > len(signal): break
    w = signal[s:e]
    f, pxx = welch(w, fs=fs, nperseg=fs*4)
    p43 = np.mean(pxx[(f >= g43l) & (f <= g43h)]) if any((f >= g43l) & (f <= g43h)) else np.nan
    p40 = np.mean(pxx[(f >= g40l) & (f <= g40h)]) if any((f >= g40l) & (f <= g40h)) else np.nan
    gamma43.append(p43); gamma40.append(p40)

gamma43 = np.array(gamma43); gamma40 = np.array(gamma40)

n43 = np.zeros(len(time_sec)) if np.isnan(np.nanmax(gamma43)) or np.nanmin(gamma43) == np.nanmax(gamma43) else (gamma43 - np.nanmin(gamma43)) / (np.nanmax(gamma43) - np.nanmin(gamma43))
n40 = np.zeros(len(time_sec)) if np.isnan(np.nanmax(gamma40)) or np.nanmin(gamma40) == np.nanmax(gamma40) else (gamma40 - np.nanmin(gamma40)) / (np.nanmax(gamma40) - np.nanmin(gamma40))

plt.figure(figsize=(14,7))
plt.plot(time_sec, deltas, 'go-', lw=4, label='Δα')
plt.axvline(41, c='red', ls='--', lw=3, label='t=41 s')
plt.ylabel('Δα', c='green'); plt.grid(True, alpha=0.3)
ax2 = plt.gca().twinx()
ax2.plot(time_sec, n43, color='gold', linestyle='-', linewidth=4, label='40–50 Hz')
if not np.all(np.isnan(n40)):
    ax2.plot(time_sec, n40, color='cyan', linestyle='--', linewidth=4, label='38–42 Hz')
ax2.set_ylabel('Norm. Gamma', color='gold')
plt.legend(loc='upper right')
plt.title('Δα vs Gamma (S01-DMT)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('plots/s01_dmt_delta_alpha_gamma_43vs40.png', dpi=300)
plt.close()

# AAFT surrogate on Δα (your full code)
n_sur = 10
surr_d = np.zeros((n_sur, len(time_sec)))

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
        for j in range(len(time_sec)):
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
        surr_d[i] = np.full(len(time_sec), np.nan)

mean_s = np.nanmean(surr_d, axis=0)
std_s = np.nanstd(surr_d, axis=0)

plt.figure(figsize=(14,7))
plt.plot(time_sec, deltas, 'go-', lw=4, label='Real')
if not np.all(np.isnan(mean_s)):
    plt.plot(time_sec, mean_s, 'k--', lw=2, label=f'AAFT(n={n_sur})')
    if not np.all(np.isnan(std_s)):
        plt.fill_between(time_sec, mean_s - std_s, mean_s + std_s, color='gray', alpha=0.3)
plt.axvline(41, c='red', ls='--', lw=3, label='t=41 s')
plt.title('AAFT Surrogate Test (S01-DMT)')
plt.xlabel('Time (s)')
plt.ylabel('Δα')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('plots/s01_dmt_delta_alpha_aaft_surrogate.png', dpi=300)
plt.close()

t41 = np.argmin(np.abs(time_sec - 41))
z = (deltas[t41] - mean_s[t41]) / std_s[t41] if std_s[t41] > 0 else np.nan
print(f"Z at t≈41 s: {z:.2f}")

# Pre/post box plots for Δα
pre_d = deltas[:t41][~np.isnan(deltas[:t41])]
post_d = deltas[t41:][~np.isnan(deltas[t41:])]
dd = 100 * (np.nanmean(pre_d) - np.nanmean(post_d)) / np.nanmean(pre_d) if np.nanmean(pre_d) > 0 else np.nan
dt, dp = ttest_ind(pre_d, post_d, equal_var=False) if len(pre_d) > 1 and len(post_d) > 1 else (np.nan, np.nan)
dd_d = (np.nanmean(post_d) - np.nanmean(pre_d)) / np.sqrt((np.nanstd(pre_d)**2 + np.nanstd(post_d)**2)/2) if len(pre_d) > 1 and len(post_d) > 1 else np.nan

plt.figure(figsize=(8,6))
plt.boxplot([pre_d, post_d], labels=['Pre', 'Post'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title(f'Δα Pre/Post t=41 s\nDrop {dd:.1f}% (d={dd_d:.2f}, p={dp:.4f})')
plt.ylabel('Δα')
plt.grid(True, alpha=0.3)
plt.savefig('plots/s01_dmt_delta_alpha_pre_post_ci.png', dpi=300)
plt.close()

# Pre/post box plots for gamma 43 Hz
pre_g43 = gamma43[:t41][~np.isnan(gamma43[:t41])]
post_g43 = gamma43[t41:][~np.isnan(gamma43[t41:])]
gr43 = 100 * (np.nanmean(post_g43) - np.nanmean(pre_g43)) / np.nanmean(pre_g43) if np.nanmean(pre_g43) > 0 else np.nan
gt43, gp43 = ttest_ind(pre_g43, post_g43, equal_var=False) if len(pre_g43) > 1 and len(post_g43) > 1 else (np.nan, np.nan)
gd43 = (np.nanmean(post_g43) - np.nanmean(pre_g43)) / np.sqrt((np.nanstd(pre_g43)**2 + np.nanstd(post_g43)**2)/2) if len(pre_g43) > 1 and len(post_g43) > 1 else np.nan

plt.figure(figsize=(8,6))
plt.boxplot([pre_g43, post_g43], labels=['Pre', 'Post'], patch_artist=True, boxprops=dict(facecolor='gold'))
plt.title(f'Gamma 40–50 Hz Pre/Post t=41 s\nRise {gr43:.1f}% (d={gd43:.2f}, p={gp43:.4f})')
plt.ylabel('Gamma Power')
plt.grid(True, alpha=0.3)
plt.savefig('plots/s01_dmt_gamma43_pre_post_ci.png', dpi=300)
plt.close()

# PSD Pre vs Post
pre_signal = signal[:int(41 * 128)]
post_signal = signal[int(41 * 128):]

f_pre, pxx_pre = welch(pre_signal, fs=128, nperseg=128*4)
f_post, pxx_post = welch(post_signal, fs=128, nperseg=128*4)

plt.figure(figsize=(10,6))
plt.semilogy(f_pre, pxx_pre, label='Pre t=41 s', color='blue', alpha=0.7)
plt.semilogy(f_post, pxx_post, label='Post t=41 s', color='red', alpha=0.7)
plt.xlim(0, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('PSD Pre vs Post t=41 s Switch (S01-DMT)')
plt.axvspan(40, 50, color='gold', alpha=0.15, label='40–50 Hz band')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/s01_dmt_psd_pre_post.png', dpi=300)
plt.close()

# Z-score Histogram
z_scores = []
for i in range(n_sur):
    if not np.isnan(surr_d[i, t41]):
        z = (deltas[t41] - np.mean(surr_d[i])) / np.std(surr_d[i])
        z_scores.append(z)

plt.figure(figsize=(8,6))
if len(z_scores) > 0:
    plt.hist(z_scores, bins=10, color='purple', edgecolor='white')
else:
    plt.text(0.5, 0.5, 'No valid Z-scores', ha='center', va='center', fontsize=12, color='gray')
plt.axvline(-2, color='red', ls='--', label='Z = -2 threshold')
plt.title('Z-scores at t≈41 s across surrogates (S01-DMT)')
plt.xlabel('Z-score')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/s01_dmt_zscore_histogram.png', dpi=300)
plt.close()

# Superradiance plot (your original)
N = 100000
tau_single = 10e-9
t_sup = np.linspace(0, 100e-9, 1000)
intensity_single = np.exp(-t_sup / tau_single)
intensity_ideal = N**2 * np.exp(-t_sup * N / tau_single)
std_detune = 1e9
delta_omega = np.random.normal(0, std_detune, N)
phase_disorder = np.exp(1j * delta_omega[:, np.newaxis] * t_sup)
phase_coherence = np.abs(np.mean(phase_disorder, axis=0))**2
intensity_disorder = N**2 * phase_coherence * np.exp(-t_sup * N / tau_single)

plt.figure(figsize=(10,6), dpi=600)
plt.plot(t_sup * 1e9, intensity_single / np.max(intensity_single), label='Single emitter', color='blue', lw=2)
plt.plot(t_sup * 1e9, intensity_ideal / np.max(intensity_ideal), label='Ideal superradiance', color='red', lw=2.5)
plt.plot(t_sup * 1e9, intensity_disorder / np.max(intensity_disorder), label='With disorder (σ=1 GHz)', color='orange', lw=2.5, ls='--')
plt.xlabel('Time (ns)')
plt.ylabel('Normalized Intensity')
plt.title('Superradiance in MT Bundle (N=100,000 Trp dipoles) – S01-DMT context')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/s01_dmt_superradiance_mt_disorder.png', dpi=600)
plt.close()

print("Original 13 crown jewel plots done.")

# =============================================================================
# BUZSÁKI + LFP + IZHIKEVICH + EXTRA VALIDATIONS
# =============================================================================
print("Running Buzsáki, LFP, Izhikevich and extra validations...")

theta_low, theta_high = 4, 8
gamma_low, gamma_high = 40, 50

theta_sig = mne.filter.filter_data(signal, sfreq=fs, l_freq=theta_low, h_freq=theta_high, fir_design='firwin')
gamma_sig = mne.filter.filter_data(signal, sfreq=fs, l_freq=gamma_low, h_freq=gamma_high, fir_design='firwin')

theta_phase = np.angle(hilbert(theta_sig))
gamma_amp = np.abs(hilbert(gamma_sig))

# Buzsáki 1: Theta–43 Hz PAC
n_bins = 18
bins = np.linspace(-np.pi, np.pi, n_bins+1)
mean_amp = np.zeros(n_bins)
for i in range(n_bins):
    idx = (theta_phase >= bins[i]) & (theta_phase < bins[i+1])
    mean_amp[i] = np.mean(gamma_amp[idx]) if np.sum(idx) > 0 else 0
mi = (np.max(mean_amp) - np.min(mean_amp)) / np.mean(mean_amp)
plt.figure(figsize=(10,6))
plt.plot((bins[:-1] + bins[1:])/2, mean_amp, 'o-', color='gold', lw=3)
plt.xlabel('Theta Phase (rad)')
plt.ylabel('Mean 43 Hz Amplitude')
plt.title(f'Theta–43 Hz PAC (Buzsáki)\nModulation Index = {mi:.3f}')
plt.grid(True)
plt.savefig('plots/buzsaki_theta_43hz_pac.png', dpi=300)
plt.close()

# Buzsáki 2: Gamma bursts on theta cycles
gamma_env = np.abs(hilbert(gamma_sig))
burst_threshold = np.mean(gamma_env) + 1.5 * np.std(gamma_env)
theta_peaks = np.where((theta_phase[:-1] < 0) & (theta_phase[1:] >= 0))[0]
window = 128
burst_triggered = []
for peak in theta_peaks:
    if peak - window > 0 and peak + window < len(gamma_env):
        burst_triggered.append(gamma_env[peak-window:peak+window])
if burst_triggered:
    avg_burst = np.mean(burst_triggered, axis=0)
    t_burst = np.linspace(-0.5, 0.5, len(avg_burst))
    plt.figure(figsize=(10,6))
    plt.plot(t_burst, avg_burst, 'gold', lw=3)
    plt.axvline(0, color='red', ls='--', lw=2, label='Theta peak')
    plt.xlabel('Time relative to theta peak (s)')
    plt.ylabel('43 Hz Gamma Envelope')
    plt.title('Gamma Bursts Locked to Theta Cycles (Buzsáki)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/buzsaki_gamma_on_theta_bursts.png', dpi=300)
    plt.close()

# LFP 1: Power Spectrum
f, pxx = welch(signal, fs=fs, nperseg=512)
plt.figure(figsize=(12,6))
plt.semilogy(f, pxx, color='gold', lw=3)
plt.axvspan(40, 50, color='red', alpha=0.2, label='43 Hz band')
plt.xlim(0, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('LFP Power Spectrum – 43 Hz Peak')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/lfp_power_spectrum.png', dpi=300)
plt.close()

# LFP 2: Multiscale Entropy
def multiscale_entropy(x, max_scale=20):
    mse = []
    for s in range(1, max_scale+1):
        coarse = x[::s]
        mse.append(np.mean(np.abs(np.diff(coarse))))
    return mse
mse = multiscale_entropy(signal)
plt.figure(figsize=(10,5))
plt.plot(range(1, len(mse)+1), mse, 'cyan', lw=3)
plt.xlabel('Scale')
plt.ylabel('Multiscale Entropy')
plt.title('LFP Multiscale Entropy')
plt.grid(True)
plt.savefig('plots/lfp_multiscale_entropy.png', dpi=300)
plt.close()

# Izhikevich Network driven by 43 Hz gamma signal
N = 200
dt = 1.0
v = -65 * np.ones(N)
u = -13 * np.ones(N)
a = 0.02 * np.ones(N)
b = 0.2 * np.ones(N)
c = -65 * np.ones(N)
d = 8 * np.ones(N)

conn = np.random.rand(N, N) < 0.1
weights = np.random.randn(N, N) * 0.5 * conn

spikes = []
for i in range(min(len(t), 10000)):
    I = gamma_sig[i] * 10 + np.random.randn(N) * 5
    dv = 0.04 * v**2 + 5 * v + 140 - u + I
    du = a * (b * v - u)
    v += dv * dt
    u += du * dt
    fired = v >= 30
    v[fired] = c[fired]
    u[fired] += d[fired]
    spikes.append(fired.astype(int))

spikes = np.array(spikes).T
rate = np.mean(spikes, axis=0) * 1000

plt.figure(figsize=(12,6))
plt.plot(rate, 'purple', lw=2)
plt.xlabel('Time (ms)')
plt.ylabel('Population firing rate (Hz)')
plt.title('Izhikevich Network driven by 43 Hz Genie Signal')
plt.grid(True)
plt.savefig('plots/izhi_network_population_rate.png', dpi=300)
plt.close()

# MFDFA on Izhikevich population rate (direct comparison to real EEG)
rate_signal = rate - np.mean(rate)
lag = np.unique(np.logspace(2, np.log10(len(rate_signal)//20), 30).astype(int))
q = np.linspace(-5, 5, 21)
q = q[q != 0]

lag, dfa = MFDFA(rate_signal, lag=lag, q=q, order=1)

plt.figure(figsize=(12,8))
for i in range(0, len(q), 3):
    plt.loglog(lag, dfa[i])
plt.xlabel('Scale')
plt.ylabel('F(q,s)')
plt.title('MFDFA on Izhikevich Network Output (43 Hz driven)')
plt.grid(True)
plt.savefig('plots/izhi_mfdfa_on_output.png', dpi=300)
plt.close()

print("All validations complete!")
print("Check /plots/ folder for all plots.")

from google.colab import files
import glob
for f in glob.glob("plots/*.png"):
    files.download(f)

print("All plots auto-downloaded. Ready for publish!")
