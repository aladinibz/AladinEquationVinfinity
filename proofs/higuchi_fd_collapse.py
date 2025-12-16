import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots folder
os.makedirs("plots", exist_ok=True)

t = np.linspace(0,120,10000)
coh = 1 - np.exp(-(t/30)**4)
fd = 1.8 - 0.6 * coh**10

plt.figure(figsize=(12,8),dpi=400)
plt.plot(t,fd,'gold',lw=4)
plt.axvline(41,color='darkred',ls='--',lw=4)
plt.title("Higuchi FD Collapse at t=41 s")
plt.xlabel("Time [s]"); plt.ylabel("FD")
plt.grid(); plt.tight_layout()

# Save with full path
save_path = "plots/higuchi_fd_collapse.png"
plt.savefig(save_path, dpi=400)
print(f"SUCCESS: Saved to {save_path}")
print("Check left Files panel → refresh → plots folder → right-click PNG → download")
plt.show()
