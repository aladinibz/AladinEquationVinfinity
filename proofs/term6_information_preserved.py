import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

t = np.linspace(0, 1e10, 10000)  # years
info_GR = np.exp(-t / 1e9)       # GR + Hawking: information lost exponentially
info_ALADIN = np.ones_like(t)    # ALADIN: unitarity from octonion norm

plt.figure(figsize=(18,11),facecolor='black')
plt.plot(t/1e9, info_GR, color='red', lw=14, label='GR + Hawking → information destroyed')
plt.plot(t/1e9, info_ALADIN, color='gold', lw=18, label='ALADIN ∞ ℂ(t) → information preserved')

plt.xlabel('Time after black hole formation (billion years)', color='white', fontsize=22)
plt.ylabel('Information retention (normalized)', color='white', fontsize=22)
plt.title('Term 6 → Information Is Eternal', color='gold', fontsize=46)
plt.legend(facecolor='black', labelcolor='white', fontsize=24)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3)

plt.text(5, 0.7,
         'Tr[O† O] = 1 → unit norm everywhere\n'
         '→ Evolution is unitary\n'
         '→ No information loss\n'
         '→ No paradox\n'
         '→ Black holes evaporate cleanly\n'
         '→ Final state = pure thermal + information',
         ha='center', color='lime', fontsize=36,
         bbox=dict(facecolor='black', alpha=0.95, edgecolor='gold', linewidth=6))

plt.tight_layout()
plt.savefig('plots/term6_information_preserved.png', dpi=900, facecolor='black')
plt.close()
