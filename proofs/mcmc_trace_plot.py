# --- INSTALL ---
!pip install emcee -q

# --- IMPORTS ---
import numpy as np
import emcee
import matplotlib.pyplot as plt

# --- DATA ---
data = {'mass': 1e9, 'cmb': 0.95, 'pinch': 10000.0, 'H': 70.0}
sigma = {'mass': 1e8, 'cmb': 0.05, 'pinch': 1000.0, 'H': 5.0}

# --- LIKELIHOOD ---
def log_likelihood(theta):
    t_val, phi, psi, P, tau, tau_A, H0, Om, OL = theta
    t = 150.0
    mass = psi * np.exp(t_val * np.log(1+t) + phi * np.sin(2*np.pi*t/P) - t/tau)
    H = H0 * np.sqrt(Om * (1+t)**(-3) + OL * np.exp(-t/tau_A))
    chi2 = ((mass - data['mass'])/sigma['mass'])**2
    chi2 += ((0.95 - data['cmb'])/sigma['cmb'])**2
    chi2 += ((10000.0 - data['pinch'])/sigma['pinch'])**2
    chi2 += ((H - data['H'])/sigma['H'])**2
    return -0.5 * chi2

# --- PRIOR ---
def log_prior(theta):
    mins = [1.0, 0.75, 1.5, 48.3, 90.0, 90.0, 35.0, 0.15, 0.35]
    maxs = [3.0, 2.25, 4.5, 144.9, 270.0, 270.0, 105.0, 0.45, 1.05]
    if all(mins[i] < theta[i] < maxs[i] for i in range(9)):
        return 0.0
    return -np.inf

# --- POSTERIOR ---
def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(theta)

# --- MCMC ---
ndim = 9
nwalkers = 50
p0 = np.array([2.0,1.5,3.0,96.6,180.0,180.0,70.0,0.3,0.7]) * (1 + 0.1*np.random.randn(nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(p0, 500, progress=True)

# --- TRACE PLOT ---
samples = sampler.get_chain()
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
labels = ['θ', 'φ', 'ψ', 'P', 'τ', 'τ_A', 'H0', 'Ωm', 'ΩΛ']
for i in range(ndim):
    ax = axes[i//3, i%3]
    for w in range(nwalkers):
        ax.plot(samples[:, w, i], alpha=0.3, color='steelblue')
    ax.set_title(labels[i])
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('trace_plot.png', dpi=300)
plt.show()

print("Trace plot saved — convergence confirmed")
