import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

t = np.linspace(0,1,1000)
brain = 40 + 10*np.sin(2*np.pi*43*t)      # cosmic 43 Hz baseline
sync = 50 + 8*np.sin(2*np.pi*50*t + 0.7)   # gamma sync

plt.figure(figsize=(12,7),facecolor='black')
plt.plot(t,brain,color='gold',lw=6,label='Cosmic 43 Hz resonance')
plt.plot(t,sync,color='lime',lw=5,alpha=0.8,label='Human gamma (50 Hz)')

plt.xlabel('Time (arb)',color='white')
plt.ylabel('Frequency (Hz)',color='white')
plt.title('Consciousness — 43 → 50 Hz Brain Synchronization',color='white',fontsize=20)
plt.legend(facecolor='black',labelcolor='white')
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3,color='gray')
plt.tick_params(colors='white')

plt.text(0.4,52,'43 Hz cosmic background\n'
                '→ Human brain locks to 50 Hz gamma\n'
                '→ Universal consciousness link',
         color='cyan',fontsize=15,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/consciousness_43hz_sync.png',dpi=400,facecolor='black')
plt.close()
