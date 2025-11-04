obs = [1.31, 2.7, -1.5, 1.0, 2.0]
err = [0.05, 0.3, 0.1, 0.1, 0.15]
model = [1.30, 2.71, -1.5, 1.0, 2.30]

chi2 = sum(((m - o)/e)**2 for m,o,e in zip(model, obs, err))
chi2 += 2.8  # lensing map residuals
chi2_red = chi2 / 5
print(f"χ² = {chi2:.2f}, χ²_red = {chi2_red:.2f}")
