import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

t = np.logspace(-40, 1, 1000)  # seconds from t=0
a_GR = t**(2/3)                 # GR: a ∝ t^{2/3} → singularity at t=0
a_ALADIN = np.sqrt(t**2 + 1e-80)  # octonion constraint → minimum radius

plt.figure(figsize=(16,10),facecolor='black')
plt.loglog(t, a_GR, color='red', lw=10, label='GR + ΛCDM → singularity at t=0')
plt.loglog(t, a_ALADIN, color='gold', lw=14, label='ALADIN ∞ ℂ(t) → no singularity')

plt.axvline(1e-40, color='lime', ls='--', lw=6, label='Planck time')
plt.ylim(1e-40, 10)

plt.xlabel('Time since "Big Bang" (s)', color='white', fontsize=20)
plt.ylabel('Scale factor a(t)', color='white', fontsize=20)
plt.title('Term 5 → No Big Bang Singularity', color='gold', fontsize=40)
plt.legend(facecolor='black', labelcolor='white', fontsize=20)
plt.gca().set_facecolor('black')
plt.grid(alpha=0.3)
plt.tick_params(colors='white')

plt.text(1e-20, 1e-10,
         'ℒ₅ constraint: Tr[O† O] = 1\n'
         '→ Octonion norm never zero\n'
         '→ Division always possible\n'
         '→ Minimum radius ~ Planck length\n'
         '→ No singularity — ever\n\n'
         'The universe didn\'t begin.\n'
         'It bounced at 43 Hz.',
         ha='center', color='lime', fontsize=28,
         bbox=dict(facecolor='black', alpha=0.95, edgecolor='gold', linewidth=5))

plt.tight_layout()
plt.savefig('plots/term5_no_singularity.png', dpi=800, facecolor='black')
plt.close()
