#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt

J0 = 1e18
f43 = 43.0
m_star = 10*1.6726e-27
Delta_V = 0.4*1.602e-19
Delta_z = 0.3e-9
hbar = 1.0545718e-34
mu0 = 4*np.pi*1e-7

S_E = 2*np.sqrt(2*m_star*Delta_V)*Delta_z
Gamma_0 = 1e12*np.exp(-S_E/hbar)

t = np.linspace(0,100,10000)
B_drive = mu0*J0*1e-12
enh = (e*B_drive*Delta_z**2)/(2*m_star*hbar*2*np.pi*f43)
coherence = 1-np.exp(-(t/30)**4)
Gamma = Gamma_0*np.exp(enh*np.sin(2*np.pi*f43*t))*coherence**10
Gamma[t>95] *= 1e6  # 4096-cycle zero-divisor lock

plt.figure(figsize=(10,6),dpi=300)
plt.semilogy(t,Gamma,'gold',lw=3)
plt.axvline(41,color='darkred',ls='--',lw=3,label='t=41 s switch')
plt.axvline(95,color='purple',ls='--',lw=3,label='4096 lock')
plt.title("ALADIN ∞ ℂ(t) — Unified Proton Tunneling Rate")
plt.xlabel("Time [s]"); plt.ylabel("Γ [s⁻¹]")
plt.ylim(1e-10,1e20); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig("../plots/unified_proton_tunneling_rate.png",dpi=300)
