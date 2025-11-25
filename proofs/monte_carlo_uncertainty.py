import numpy as np
import matplotlib.pyplot as plt

# Base parameters
base = {
    'theta': 2.0, 'phi': 1.5, 'psi': 3.0, 'P': 96.6, 'tau': 180.0,
    'tau_A': 180.0, 'H0': 70.0, 'Om': 0.3, 'OL': 0.7
}

def aladin_mass(t, p):
    return p['psi'] * np.exp(p['theta'] * np.log(1+t) + p['phi'] * np.sin(2*np.pi*t/p['P']) - t/p['tau'])

def H_Maria(t, p):
    return p['H0'] * np.sqrt(p['Om'] * (1+t)**(-3) + p['OL'] * np.exp(-t/p['tau_A']))

# Monte Carlo
n_runs = 10000
t_150 = 150
results = {'mass': [], 'H': []}

for _ in range(n_runs):
    p = {k: np.random.normal(v, 0.2*v) for k, v in base.items()}
    results['mass'].append(aladin_mass(t_150, p))
    results['H'].append(H_Maria(0, p))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(results['mass'], bins=50, color='purple', alpha=0.7)
ax1.set_title('Mass @150 Myr (10k runs)')
ax1.set_xlabel('M⊙')
ax2.hist(results['H'], bins=50, color='gold', alpha=0.7)
ax2.set_title('H_Maria(t=0) (10k runs)')
ax2.set_xlabel('km/s/Mpc')
plt.tight_layout()
plt.savefig('monte_carlo_uncertainty.png', dpi=300)

print(f"Mass: {np.mean(results['mass']):.2e} ± {np.std(results['mass']):.2e} M⊙")
print(f"H: {np.mean(results['H']):.1f} ± {np.std(results['H']):.1f} km/s/Mpc")
