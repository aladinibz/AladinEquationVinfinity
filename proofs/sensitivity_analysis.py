import numpy as np
import matplotlib.pyplot as plt

# Base parameters
params = {
    'theta': 2.0,
    'phi': 1.5,
    'psi': 3.0,
    'P': 96.6,
    'tau': 180.0,
    'tau_A': 180.0,
    'H0': 70.0,
    'Om': 0.3,
    'OL': 0.7
}

def aladin_mass(t, p):
    return p['psi'] * np.exp(p['theta'] * np.log(1+t) + p['phi'] * np.sin(2*np.pi*t/p['P']) - t/p['tau'])

def H_Maria(t, p):
    return p['H0'] * np.sqrt(p['Om'] * (1+t)**(-3) + p['OL'] * np.exp(-t/p['tau_A']))

# Time points
t_150 = 150
t_now = 0

# Sensitivity sweep
results = {}
for key in params:
    base = params[key]
    sweep = np.linspace(0.5*base, 1.5*base, 50)
    mass = [aladin_mass(t_150, {**params, key:v}) for v in sweep]
    H_now = [H_Maria(t_now, {**params, key:v}) for v in sweep]
    results[key] = {'sweep': sweep, 'mass': mass, 'H': H_now}

# Plot
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()
for i, key in enumerate(params):
    ax = axes[i]
    ax.plot(results[key]['sweep'], results[key]['mass'], 'purple', label='Mass @150Myr')
    ax.plot(results[key]['sweep'], results[key]['H'], 'gold', label='H(t=0)')
    ax.set_title(f'{key} sensitivity')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=300)

print("Sensitivity analysis complete — saved as sensitivity_analysis.png")
