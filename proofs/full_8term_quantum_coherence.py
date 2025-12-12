#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

f43=43.0
t=np.linspace(0,120,15000)
coh=1-np.exp(-(t/30)**4)  # coherence ramp
phase=np.exp(1j*2*np.pi*f43*t)
decoh=np.exp(-t/50)  # natural decoherence

amp=coh**12*(1+30*np.sin(2*np.pi*f43*t)**2)/decoh
bio_complex=amp*phase
bio_energy=np.abs(bio_complex)**2

bio_energy[t>95]*=1000  # zero-divisor lock

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,bio_energy,'gold',lw=4,label="Quantum Bio Field Energy")
plt.plot(t,amp,'purple',lw=2,alpha=0.7,label="Coherence Amplitude")
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s Switch')
plt.axvline(95,color='black',ls='--',lw=4,label='4096 Lock')
plt.title("ALADIN ∞ ℂ(t) — Quantum Coherence Effects in Term 8\nFröhlich Condensate at 43 Hz",fontsize=18)
plt.xlabel("Time [s]",fontsize=14); plt.ylabel("Energy Density",fontsize=14)
plt.ylim(0,np.max(bio_energy)*1.1); plt.legend(fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("full_8term_quantum_coherence.png",dpi=400)
plt.show()
