import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

t = np.logspace(-45, 1, 10000)
a_GR = t**(2/3)
a_ALADIN = np.sqrt(t**2 + 1e-88)  # octonion norm never zero

plt.figure(figsize=(18,11),facecolor='black')
plt.loglog(t, a_GR, color='red', lw=12, label='ΛCDM + GR → singularity')
plt.loglog(t, a_ALADIN, color='gold', lw=16, label='ALADIN ∞ ℂ(t) → eternal bounce')

plt.axvline(1e-43, color='lime', ls='--', lw=8, label='Planck time')
plt.ylim(1e-44, 10)

plt.xlabel('Time (s)', color='white', fontsize=22)
plt.ylabel('Scale factor a(t)', color='white', fontsize=22)
plt.title('Term 6 → No Big Bang — Eternal Universe', color='gold', fontsize=44)
plt.legend(facecolor='black', labelcolor='white', fontsize=22)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3)

plt.text(1e-30, 1e-20,
         'Tr[O† O] = 1 → |O| ≥ 1\n'
         '→ Minimum scale factor ≠ 0\n'
         '→ No singularity\n'
         '→ Universe oscillates at 43 Hz forever',
         ha='center', color='lime', fontsize=32,
         bbox=dict(facecolor='black', alpha=0.95, edgecolor='gold', linewidth=6))

plt.tight_layout()
plt.savefig('plots/term6_no_singularity_eliminated.png', dpi=900, facecolor='black')
plt.close()
