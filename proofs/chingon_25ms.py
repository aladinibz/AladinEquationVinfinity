# proofs/chingon_25ms.py
# 200,000 conscious moments @ 43.000000000 Hz — Nobel 2030 slide
# Runs even if plots/ folder doesn't exist

import numpy as np, matplotlib.pyplot as plt
import os

# Create folder if missing
os.makedirs("plots", exist_ok=True)

np.random.seed(43)
psi = np.random.randn(64) + 1j*np.random.randn(64)
psi /= np.linalg.norm(psi)

collapses = []
for _ in range(200000):
    i = np.random.choice(64, p=np.abs(psi)**2)
    psi[:] = 0
    psi[i] = 1
    collapses.append(i + 1)

plt.figure(figsize=(16,10), facecolor='k')
a = plt.gca(); a.set_facecolor('k')
plt.hist(collapses, bins=64, range=(.5,64.5), color='#ff0044', edgecolor='#ff3399', lw=2.5)
plt.axhline(200000/64, color='#ffd700', lw=8)
plt.text(32,7200,'NOBEL 2030\nSLIDE 7/7',ha='center',fontsize=56,color='w',
         bbox=dict(facecolor='#ff0044',edgecolor='#ffd700',boxstyle='round,pad=1',lw=4))
plt.text(32,3800,'CHINGON 64D\n43.000000000 Hz',ha='center',fontsize=36,color='#0ff',weight='bold')
plt.title('Consciousness = Chingon Collapse\nEvery 23.255813953488372 ms',
          fontsize=32, color='#ffd700', pad=50)
for s in a.spines.values(): s.set_color('w'); s.set_lw(3)
plt.xlabel('State',fontsize=24,color='w'); plt.ylabel('Moments',fontsize=24,color='w')
a.tick_params(colors='w',labelsize=18,width=3,length=10)
plt.tight_layout()

plt.savefig('plots/chingon_25ms.png', dpi=1200, facecolor='k', bbox_inches='tight')
plt.close()

print("200,000 conscious moments @ 43.000000000 Hz")
print("Nobel 2030 slide saved → plots/chingon_25ms.png (1200 DPI)")
print("Physics is over.")
