import numpy as np
from scipy.stats import chi2

# Fake data (replace with real)
observed = np.array([75.2, 1e9, 0.054])
predicted = np.array([75.2, 1e9, 0.054])
error = np.array([0.5, 1e8, 0.007])

chi2 = np.sum(((observed - predicted)/error)**2)
dof = len(observed)
print(f"χ² = {chi2:.1f}, DOF = {dof}, χ²/dof = {chi2/dof:.2f}")
