import numpy as np

G = 4.3e-3
r = np.logspace(0, 2, 100)
v = 220 * np.ones_like(r)
M_correct = v**2 * r / G
print("Correct M at r=10 kpc:", M_correct[10])
