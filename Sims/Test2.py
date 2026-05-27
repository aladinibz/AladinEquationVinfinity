
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

np.random.seed(42)
pert = 0.08
Bx[1:-1] = cp.asarray(np.random.randn(N-1, N, N) * pert, dtype=cp.float32)
By[:,1:-1] = cp.asarray(np.random.randn(N, N-1, N) * pert, dtype=cp.float32)
Bz[:-1,:-1] = cp.asarray(np.random.randn(N-1, N-1, N+1) * pert * 0.5 + 0.5, dtype=cp.float32)

print("✅ Shapes OK")

# ====================== CORRECT TEXTURE BINDING ======================
def create_texture(arr):
    # Create CUDA array
    cuda_array = cp.cuda.texture.CUDAarray(arr, arr.shape, arr.dtype)
    
    # Resource Descriptor
    res_desc = cp.cuda.texture.ResourceDescriptor(
        cp.cuda.texture.ResourceType.CUDA_ARRAY,
        cuArr=cuda_array
    )
    
    # Texture Descriptor
    tex_desc = cp.cuda.texture.TextureDescriptor(
        addressMode=cp.cuda.texture.AddressMode.Clamp,
        filterMode=cp.cuda.texture.FilterMode.Point,
        normalizedCoords=False
    )
    
    return cp.cuda.texture.TextureObject(res_desc, tex_desc)

# Create textures
tex_vx = create_texture(mx / rho)
tex_vy = create_texture(my / rho)
tex_vz = create_texture(mz / rho)

tex_Bx = create_texture(Bx)
tex_By = create_texture(By)
tex_Bz = create_texture(Bz)

print("✅ Textures created successfully with correct CuPy syntax!")
