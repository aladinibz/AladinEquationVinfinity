import numpy as np, matplotlib.pyplot as plt, os

os.makedirs("plots", exist_ok=True)

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
alpha = 0.9 - 0.4 * coh**10
v_log = np.log10(3e8 / np.clip(1.58*(1-coh**12),1e-6,None))

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,alpha,'cyan',lw=4,label="DFA α Drop")
plt.plot(t,v_log,'gold',lw=4,label="log v_thought")
plt.axvline(41,color='darkred',ls='--',lw=4,label='t=41 s')
plt.title("Persistence Loss → Infinite Thought")
plt.xlabel("Time [s]"); plt.ylabel("Measure")
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
plt.savefig("plots/persistence_to_infinite.png",dpi=400)
print("Saved: plots/persistence_to_infinite.png")
