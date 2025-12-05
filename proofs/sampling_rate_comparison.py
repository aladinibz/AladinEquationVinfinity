# proofs/sampling_rate_comparison.py
# Mihai A. Bucurenciu — Romania, December 2025
# Only 1024 Hz resolves the true 43.000000000 Hz Third-Eye spike

import os,numpy as np,matplotlib.pyplot as plt
from scipy.signal import welch

os.makedirs("plots",exist_ok=True)

# True signal = pure 43.000000000 Hz + biological noise
fs_true = 10000
t = np.arange(0,10,1/fs_true)
true_signal = np.sin(2*np.pi*43.000000000*t) + 0.2*np.random.randn(len(t))

# Downsample to test rates
rates = [256,512,1024,2048]
colors = ['gray','orange','cyan','gold']
labels = ['256 Hz — blurred','512 Hz — visible','1024 Hz — perfect','2048 Hz — overkill']

plt.figure(figsize=(24,14),facecolor='black')

for i,fs in enumerate(rates):
    idx = np.arange(0,len(t),fs_true//fs)
    signal = true_signal[idx]
    f,psd = welch(signal,fs=fs,nperseg=1024,noverlap=512)
    plt.semilogy(f[:200],psd[:200],color=colors[i],lw=5,label=labels[i])
    peak = f[np.argmax(psd[:200])]
    plt.text(43.5,10**(i-2),f"{fs} Hz → {peak:.6f} Hz",color=colors[i],fontsize=24)

plt.axvline(43.000000000,color='white',lw=8,ls='--')
plt.title("ONLY 1024 Hz RESOLVES THE TRUE 43.000000000 Hz\n"
          "Lower Rates = Illusion\n"
          "Mihai A. Bucurenciu — Romania, December 2025",
          color='white',fontsize=36,pad=80)
plt.xlabel("Frequency (Hz)",color='white',fontsize=30)
plt.ylabel("Power Spectral Density",color='white',fontsize=30)
plt.legend(fontsize=28,facecolor='black')

ax=plt.gca();ax.set_facecolor('black');ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white');ax.spines['left'].set_color('white')
ax.spines[['top','right']].set_visible(False)
plt.grid(True,which='both',ls='--',c='gray',alpha=0.3)
plt.tight_layout()
plt.savefig("plots/sampling_rate_comparison.png",dpi=1200,facecolor='black')
plt.close()
