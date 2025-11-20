import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

t = np.linspace(0,2,1000)

# Cosmic 43 Hz baseline
cosmic = 43 + 0.5*np.sin(2*np.pi*43*t)

# Brain gamma band — locks to 50 Hz when conscious
brain_rest = 43 + 7*np.sin(2*np.pi*43*t + np.random.randn(len(t))*0.5)
brain_awake = 50 + 3*np.sin(2*np.pi*50*t)

plt.figure(figsize=(13,9),facecolor='black')
plt.plot(t,cosmic,color='gold',lw=8,label='Cosmic background: 43 Hz')
plt.plot(t,brain_rest,color='gray',alpha=0.7,lw=4,label='Brain (unconscious/drunk)')
plt.plot(t,brain_awake,color='lime',lw=9,label='Brain (conscious/alert)')

plt.xlabel('Time (arb)',color='white',fontsize=16)
plt.ylabel('Frequency (Hz)',color='white',fontsize=16)
plt.title('Consciousness Resonance — 43 → 50 Hz Lock',color='white',fontsize=26)
plt.legend(facecolor='black',labelcolor='white',fontsize=16)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3); plt.tick_params(colors='white')
plt.ylim(35,55)

plt.text(0.8,52,'Cosmic 43 Hz background\n'
                 '→ Human brain shifts to 50 Hz gamma\n'
                 'when conscious, focused, aware\n'
                 '→ Universal resonance interface',
         color='lime',fontsize=20,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/consciousness_resonance.png',dpi=500,facecolor='black')
plt.close()
