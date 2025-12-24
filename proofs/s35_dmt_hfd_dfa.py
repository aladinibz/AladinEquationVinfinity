import mne
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

raw = mne.io.read_raw_bdf('/content/S35-DMT.bdf', preload=True)
raw.filter(1, 100); raw.notch_filter(np.arange(50, 251, 50))

picks = mne.pick_types(raw.info, eeg=True)[:1]
data, times = raw[picks, :]

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

def dfa(ts):
    ts = np.asarray(ts).flatten()
    y = np.cumsum(ts - ts.mean())
    scales = np.logspace(np.log10(4), np.log10(len(ts)//4), 15, dtype=int)
    F_n = []
    for n in scales:
        segments = len(ts) // n
        rms = []
        for v in range(segments):
            seg = y[v*n:(v+1)*n]
            coeffs = np.polyfit(np.arange(n), seg, 1)
            trend = np.polyval(coeffs, np.arange(n))
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        F_n.append(np.mean(rms))
    return np.polyfit(np.log(scales), np.log(F_n), 1)[0]

hfd_dmt = higuchi_fd(data[0])
dfa_dmt = dfa(data[0])
print(f"S35-DMT: HFD = {hfd_dmt:.4f}, DFA α = {dfa_dmt:.4f}")

plt.figure(figsize=(12,8),dpi=1200)
plt.bar(['HFD', 'DFA α'], [hfd_dmt, dfa_dmt], color='gold')
plt.title("Subject 35 — DMT Breakthrough\nHFD + DFA α",fontsize=18)
plt.ylabel("Value")
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/s35_dmt_hfd_dfa.png",dpi=1200)
plt.show()
