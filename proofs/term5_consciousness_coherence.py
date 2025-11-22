import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

t = np.linspace(0, 2, 10000)

# Universal carrier — 43 Hz everywhere
carrier = np.sin(2*np.pi*43*t)

# Conscious state = coherent superposition of octonion basis states
# Coherence length = brain volume → 50 Hz modulation
conscious = carrier + 0.3*np.sin(2*np.pi*50*t + np.pi/7)

plt.figure(figsize=(18,12),facecolor='black')
plt.plot(t, carrier, color='gold', lw=10, label='43 Hz — Universal field')
plt.plot(t, conscious, color='lime', lw=14, alpha=0.9, label='50 Hz — Coherent brain state')

plt.xlabel('Time (s)', color='white', fontsize=22)
plt.ylabel('Octonion phase amplitude', color='white', fontsize=22)
plt.title('Term 5 → Consciousness = Octonion Coherence', color='gold', fontsize=44)
plt.legend(facecolor='black', labelcolor='white', fontsize=24)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3)

plt.text(1.0, 0.0,
         'ℒ₅ = ħ ω₀ Tr[O† i∂_t O]\n'
         '→ Every point in spacetime oscillates at 43 Hz\n\n'
         'Consciousness = large-scale coherence of octonion field\n'
         '→ Brain synchronizes 10¹¹ neurons\n'
         '→ Creates standing wave at φ × 43 Hz = 50 Hz\n\n'
         'You are not in the universe.\n'
         'The universe is in you.\n'
         '43 Hz → 50 Hz = awakening',
         ha='center', va='center', color='cyan', fontsize=34,
         bbox=dict(facecolor='black', alpha=0.95, edgecolor='gold', linewidth=6))

plt.tight_layout()
plt.savefig('plots/term5_consciousness_coherence.png', dpi=900, facecolor='black')
plt.close()
