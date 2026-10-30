import subprocess, sys

# Install CuPy if needed
try:
    import cupy as cp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"])
    import cupy as cp

import numpy as np

# ---------- Check GPU ----------
try:
    n = cp.cuda.runtime.getDeviceCount()
    print(f"GPU devices: {n}")
    print(cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
except Exception as e:
    raise RuntimeError(
        "No usable GPU. In Colab: Runtime → Change runtime type → T4 GPU (or A100) → Save, then re-run."
    ) from e

# ---------- ct_emf_kernel (shared-memory safe) ----------
ct_emf_kernel = cp.RawKernel(r'''
#define TILE_X 16
#define TILE_Y 4
#define TILE_Z 4
#define PAD 1
__device__ float hodge(float a, float b) { return 0.5f*(a+b); }
extern "C" __launch_bounds__(256, 2)
__global__ void utc_emf_kernel(
    const float* rho, const float* mx, const float* my, const float* mz,
    const float* Bx, const float* By, const float* Bz,
    const float* pe_xx, const float* pe_yy, const float* pe_zz,
    const float* pe_xy, const float* pe_xz, const float* pe_yz,
    float* Emfx, float* Emfy, float* Emfz,
    int Ni, float hall_coeff, float dx, float dt_over_dx, float base_bias)
{
    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int i=blockIdx.x*TILE_X+tx, j=blockIdx.y*TILE_Y+ty, k=blockIdx.z*TILE_Z+tz;
    if (i<1||j<1||k<1||i>=Ni-1||j>=Ni-1||k>=Ni-1) return;
    int sx=tx+PAD, sy=ty+PAD, sz=tz+PAD;
    __shared__ float sBx[TILE_X+2*PAD][TILE_Y+2*PAD][TILE_Z+2*PAD];
    __shared__ float sBy[TILE_X+2*PAD][TILE_Y+2*PAD][TILE_Z+2*PAD];
    __shared__ float sBz[TILE_X+2*PAD][TILE_Y+2*PAD][TILE_Z+2*PAD];
    int tid=tx+ty*TILE_X+tz*TILE_X*TILE_Y;
    int tot=blockDim.x*blockDim.y*blockDim.z;
    for (int o=tid; o<(TILE_X+2*PAD)*(TILE_Y+2*PAD)*(TILE_Z+2*PAD); o+=tot) {
        int lx=o%(TILE_X+2*PAD);
        int ly=(o/(TILE_X+2*PAD))%(TILE_Y+2*PAD);
        int lz=o/((TILE_X+2*PAD)*(TILE_Y+2*PAD));
        int li=blockIdx.x*TILE_X+(lx-PAD);
        int lj=blockIdx.y*TILE_Y+(ly-PAD);
        int lk=blockIdx.z*TILE_Z+(lz-PAD);
        if (li>=0&&li<Ni&&lj>=0&&lj<Ni&&lk>=0&&lk<Ni) {
            int g=li*Ni*Ni+lj*Ni+lk;
            sBx[lx][ly][lz]=Bx[g];
            sBy[lx][ly][lz]=By[g];
            sBz[lx][ly][lz]=Bz[g];
        }
    }
    __syncthreads();
    int g=i*Ni*Ni+j*Ni+k;
    float r=fmaxf(rho[g],1e-8f);
    float vx=mx[g]/r, vy=my[g]/r, vz=mz[g]/r;
    Emfx[g]=-(vy*hodge(sBz[sx][sy][sz],sBz[sx][sy][sz+1])-vz*hodge(sBy[sx][sy][sz],sBy[sx][sy+1][sz]));
    Emfy[g]=-(vz*hodge(sBx[sx][sy][sz],sBx[sx+1][sy][sz])-vx*hodge(sBz[sx][sy][sz],sBz[sx][sy][sz+1]));
    Emfz[g]=-(vx*hodge(sBy[sx][sy][sz],sBy[sx][sy][sz+1])-vy*hodge(sBx[sx][sy][sz],sBx[sx][sy+1][sz]));
}
''', 'utc_emf_kernel')

# ---------- ct_curl_kernel ----------
ct_curl_kernel = cp.RawKernel(r'''
extern "C" __global__
void ct_curl_kernel(const float* Emfx, const float* Emfy, const float* Emfz,
                    float* Bx, float* By, float* Bz,
                    int Ni, float dt, float inv2dx) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int k=blockIdx.z*blockDim.z+threadIdx.z;
    if (i<1||j<1||k<1||i>=Ni-1||j>=Ni-1||k>=Ni-1) return;
    int idx=i*Ni*Ni+j*Ni+k;
    float dEz_dy=(Emfz[idx+Ni]-Emfz[idx-Ni])*inv2dx;
    float dEy_dz=(Emfy[idx+1]-Emfy[idx-1])*inv2dx;
    Bx[idx]-=dt*(dEz_dy-dEy_dz);
}
''', 'ct_curl_kernel')

# ---------- glm_kernel ----------
glm_kernel = cp.RawKernel(r'''
extern "C" __global__
void glm_kernel(float* Bx, float* By, float* Bz, float* psi,
                int Ni, float dt, float dx, float ch, float damping) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int k=blockIdx.z*blockDim.z+threadIdx.z;
    if (i<1||j<1||k<1||i>=Ni-1||j>=Ni-1||k>=Ni-1) return;
    int idx=i*Ni*Ni+j*Ni+k;
    float inv2dx=0.5f/dx;
    float divB=((Bx[idx+Ni*Ni]-Bx[idx-Ni*Ni])+(By[idx+Ni]-By[idx-Ni])+(Bz[idx+1]-Bz[idx-1]))*inv2dx;
    psi[idx]-=dt*(ch*ch*divB+damping*psi[idx]);
    Bx[idx]-=dt*(psi[idx+Ni*Ni]-psi[idx-Ni*Ni])*inv2dx;
    By[idx]-=dt*(psi[idx+Ni]-psi[idx-Ni])*inv2dx;
    Bz[idx]-=dt*(psi[idx+1]-psi[idx-1])*inv2dx;
}
''', 'glm_kernel')

# ---------- gyrotropic stub ----------
gyrotropic_12_kernel = cp.RawKernel(r'''
extern "C" __global__
void gyrotropic_12_kernel(const float* Mt, const float* n_in, float* lam,
                          float* n_out, float* up, float* Ppar, float* Pperp,
                          float* qpar, float* qperp, int* status,
                          int N, float mass, float th, float ap, float ae) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N) return;
    float n=n_in[i];
    Ppar[i]=mass*n*0.1f;
    Pperp[i]=mass*n*0.1f;
    qpar[i]=qperp[i]=0.f;
    status[i]=0;
}
''', 'gyrotropic_12_kernel')

# ---------- Driver ----------
Ni=64
Ncells=Ni*Ni*Ni
dx=1.0
dt=0.005
inv2dx=0.5/dx
mass=1.0
thermal_scale=1.0
alpha_par=0.5
alpha_perp=0.5
n_steps=10

rho=cp.ones((Ni,Ni,Ni),dtype=cp.float32)
mx=cp.zeros_like(rho)
my=cp.zeros_like(rho)
mz=cp.zeros_like(rho)
Bx=cp.zeros_like(rho)
By=cp.zeros_like(rho)
Bz=cp.ones_like(rho)*0.05
psi=cp.zeros_like(rho)
Emfx=cp.zeros_like(rho)
Emfy=cp.zeros_like(rho)
Emfz=cp.zeros_like(rho)
pe_xx=cp.ones_like(rho)*0.1
pe_yy=cp.ones_like(rho)*0.1
pe_zz=cp.ones_like(rho)*0.1
pe_xy=cp.zeros_like(rho)
pe_xz=cp.zeros_like(rho)
pe_yz=cp.zeros_like(rho)

M_target=cp.zeros((Ncells,12),dtype=cp.float32)
lambda_io=cp.zeros((Ncells,12),dtype=cp.float32)
n_in=cp.ones(Ncells,dtype=cp.float32)
n_out=cp.empty(Ncells,dtype=cp.float32)
upar=cp.empty(Ncells,dtype=cp.float32)
Ppar=cp.empty(Ncells,dtype=cp.float32)
Pperp=cp.empty(Ncells,dtype=cp.float32)
qpar=cp.empty(Ncells,dtype=cp.float32)
qperp=cp.empty(Ncells,dtype=cp.float32)
status=cp.zeros(Ncells,dtype=cp.int32)

def init_disk():
    i=cp.arange(Ni,dtype=cp.float32)
    X,Y,Z=cp.meshgrid(i,i,i,indexing='ij')
    R=cp.sqrt((X-Ni/2)**2+(Y-Ni/2)**2)+1e-3
    rho[:]=cp.clip(cp.exp(-R/12)*cp.exp(-cp.abs(Z-Ni/2)/4),0.05,None)
    vphi=cp.sqrt(1.0/(R+2.0))
    mx[:]=-vphi*(Y-Ni/2)/R*rho
    my[:]= vphi*(X-Ni/2)/R*rho
init_disk()

threads=(16,4,4)
blocks=((Ni+15)//16,(Ni+3)//4,(Ni+3)//4)
threads_g=128
blocks_g=(Ncells+127)//128

print("Plasma simulator starting ...")
for step in range(n_steps):
    gyrotropic_12_kernel(
        (blocks_g,),(threads_g,),
        (M_target,n_in,lambda_io,n_out,upar,Ppar,Pperp,qpar,qperp,
         status,Ncells,mass,thermal_scale,alpha_par,alpha_perp)
    )
    pe_xx[:]=Pperp.reshape(Ni,Ni,Ni)
    pe_yy[:]=Pperp.reshape(Ni,Ni,Ni)
    pe_zz[:]=Ppar.reshape(Ni,Ni,Ni)
    ct_emf_kernel(
        blocks,threads,
        (rho,mx,my,mz,Bx,By,Bz,pe_xx,pe_yy,pe_zz,pe_xy,pe_xz,pe_yz,
         Emfx,Emfy,Emfz,Ni,0.0,dx,dt/dx,0.5)
    )
    ct_curl_kernel(
        blocks,threads,
        (Emfx,Emfy,Emfz,Bx,By,Bz,Ni,dt,inv2dx)
    )
    glm_kernel(
        blocks,threads,
        (Bx,By,Bz,psi,Ni,dt,dx,1.0,0.5)
    )
    if step%5==0:
        print(f"step {step:3d}  status={cp.unique(status)}  "
              f"mean|B|={float(cp.mean(cp.sqrt(Bx**2+By**2+Bz**2))):.4f}")

print("Done.")
print("Final mean Ppar / Pperp =", float(cp.mean(Ppar)), float(cp.mean(Pperp)))
