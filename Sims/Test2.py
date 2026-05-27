import numpy as np
import cupy as cp

# ====================== PARAMETERS ======================
N = 256
L = 1.0
dx = L / N

# ====================== DATA ON GPU ======================
rho = cp.ones((N, N, N), dtype=cp.float32)
mx = cp.zeros((N, N, N), dtype=cp.float32)
my = cp.zeros((N, N, N), dtype=cp.float32)
mz = cp.zeros((N, N, N), dtype=cp.float32)
E_total = cp.ones((N, N, N), dtype=cp.float32) * 3.0

Bx = cp.zeros((N+1, N, N), dtype=cp.float32)
By = cp.zeros((N, N+1, N), dtype=cp.float32)
Bz = cp.zeros((N, N, N+1), dtype=cp.float32)

# ====================== FIXED INITIAL CONDITION ======================
np.random.seed(42)
pert = 0.08

Bx[1:-1] = cp.asarray(np.random.randn(N-1, N, N) * pert, dtype=cp.float32)
By[:,1:-1] = cp.asarray(np.random.randn(N, N-1, N) * pert, dtype=cp.float32)
Bz[:-1,:-1] = cp.asarray(np.random.randn(N-1, N-1, N+1) * pert * 0.5 + 0.5, dtype=cp.float32)

print("✅ Shapes fixed:")
print("Bx shape:", Bx.shape)
print("By shape:", By.shape)
print("Bz shape:", Bz.shape)

# ====================== TEXTURE BINDING ======================
tex_vx = cp.cuda.texture.Texture3D(mx / rho, cp.cuda.texture.ResampleMode.Point)
tex_vy = cp.cuda.texture.Texture3D(my / rho, cp.cuda.texture.ResampleMode.Point)
tex_vz = cp.cuda.texture.Texture3D(mz / rho, cp.cuda.texture.ResampleMode.Point)

tex_Bx = cp.cuda.texture.Texture3D(Bx, cp.cuda.texture.ResampleMode.Point)
tex_By = cp.cuda.texture.Texture3D(By, cp.cuda.texture.ResampleMode.Point)
tex_Bz = cp.cuda.texture.Texture3D(Bz, cp.cuda.texture.ResampleMode.Point)

print("✅ Textures created successfully.")
