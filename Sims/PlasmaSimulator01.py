import subprocess, sys
try:
    import cupy as cp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"])
    import cupy as cp
import numpy as np

# GPU check
try:
    assert cp.cuda.runtime.getDeviceCount() > 0
    print("GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
except Exception:
    raise RuntimeError("Enable T4/A100 GPU: Runtime → Change runtime type → GPU")

# ---------- ct_emf ----------
ct_emf_kernel = cp.RawKernel(r'''
#define TX 16
#define TY 4
#define TZ 4
#define PAD 1
__device__ float hodge(float a, float b){return 0.5f*(a+b);}
extern "C" __launch_bounds__(256,2)
__global__ void utc_emf_kernel(
    const float* rho, const float* mx, const float* my, const float* mz,
    const float* Bx, const float* By, const float* Bz,
    float* Emfx, float* Emfy, float* Emfz, int Ni)
{
    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    int i=blockIdx.x*TX+tx, j=blockIdx.y*TY+ty, k=blockIdx.z*TZ+tz;
    if(i<1||j<1||k<1||i>=Ni-1||j>=Ni-1||k>=Ni-1) return;
    int sx=tx+PAD, sy=ty+PAD, sz=tz+PAD;
    __shared__ float sBx[TX+2*PAD][TY+2*PAD][TZ+2*PAD];
    __shared__ float sBy[TX+2*PAD][TY+2*PAD][TZ+2*PAD];
    __shared__ float sBz[TX+2*PAD][TY+2*PAD][TZ+2*PAD];
    int tid=tx+ty*TX+tz*TX*TY;
    int tot=blockDim.x*blockDim.y*blockDim.z;
    for(int o=tid;o<(TX+2*PAD)*(TY+2*PAD)*(TZ+2*PAD);o+=tot){
        int lx=o%(TX+2*PAD);
        int ly=(o/(TX+2*PAD))%(TY+2*PAD);
        int lz=o/((TX+2*PAD)*(TY+2*PAD));
        int li=blockIdx.x*TX+(lx-PAD);
        int lj=blockIdx.y*TY+(ly-PAD);
        int lk=blockIdx.z*TZ+(lz-PAD);
        if(li>=0&&li<Ni&&lj>=0&&lj<Ni&&lk>=0&&lk<Ni){
            int g=li*Ni*Ni+lj*Ni+lk;
            sBx[lx][ly][lz]=Bx[g];
            sBy[lx][ly][lz]=By[g];
            sBz[lx][ly][lz]=Bz[g];
        } else {
            sBx[lx][ly][lz]=sBy[lx][ly][lz]=sBz[lx][ly][lz]=0.f;
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
''','utc_emf_kernel')

# ---------- ct_curl (three-component) ----------
ct_curl_kernel = cp.RawKernel(r'''
extern "C" __global__
void ct_curl_kernel(
    const float* Emfx, const float* Emfy, const float* Emfz,
    float* Bx, float* By, float* Bz,
    int Ni, float dt, float inv2dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || j < 1 || k < 1 || i >= Ni-1 || j >= Ni-1 || k >= Ni-1) return;
    int idx = i * Ni * Ni + j * Ni + k;

    float dEz_dy = (Emfz[idx + Ni] - Emfz[idx - Ni]) * inv2dx;
    float dEy_dz = (Emfy[idx + 1]  - Emfy[idx - 1])  * inv2dx;
    Bx[idx] -= dt * (dEz_dy - dEy_dz);

    float dEx_dz = (Emfx[idx + 1]     - Emfx[idx - 1])     * inv2dx;
    float dEz_dx = (Emfz[idx + Ni*Ni] - Emfz[idx - Ni*Ni]) * inv2dx;
    By[idx] -= dt * (dEx_dz - dEz_dx);

    float dEy_dx = (Emfy[idx + Ni*Ni] - Emfy[idx - Ni*Ni]) * inv2dx;
    float dEx_dy = (Emfx[idx + Ni]    - Emfx[idx - Ni])    * inv2dx;
    Bz[idx] -= dt * (dEy_dx - dEx_dy);
}
''', 'ct_curl_kernel')

# ---------- glm ----------
glm_kernel = cp.RawKernel(r'''
extern "C" __global__
void glm_kernel(float* Bx,float* By,float* Bz,float* psi,
                int Ni,float dt,float dx,float ch,float damping){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int k=blockIdx.z*blockDim.z+threadIdx.z;
    if(i<1||j<1||k<1||i>=Ni-1||j>=Ni-1||k>=Ni-1) return;
    int idx=i*Ni*Ni+j*Ni+k;
    float inv2dx=0.5f/dx;
    float divB=((Bx[idx+Ni*Ni]-Bx[idx-Ni*Ni])+(By[idx+Ni]-By[idx-Ni])+(Bz[idx+1]-Bz[idx-1]))*inv2dx;
    psi[idx]-=dt*(ch*ch*divB+damping*psi[idx]);
    Bx[idx]-=dt*(psi[idx+Ni*Ni]-psi[idx-Ni*Ni])*inv2dx;
    By[idx]-=dt*(psi[idx+Ni]-psi[idx-Ni])*inv2dx;
    Bz[idx]-=dt*(psi[idx+1]-psi[idx-1])*inv2dx;
}
''','glm_kernel')

# ---------- gyrotropic stub ----------
gyrotropic_12_kernel = cp.RawKernel(r'''
extern "C" __global__
void gyrotropic_12_kernel(const float* Mt,const float* n_in,float* lam,
                          float* n_out,float* up,float* Ppar,float* Pperp,
                          float* qpar,float* qperp,int* status,
                          int N,float mass,float th,float ap,float ae){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    float n=n_in[i];
    Ppar[i]=mass*n*0.1f;
    Pperp[i]=mass*n*0.1f;
    qpar[i]=qperp[i]=0.f;
    status[i]=0;
}
''','gyrotropic_12_kernel')

# ---------- Driver ----------
Ni=64
Ncells=Ni*Ni*Ni
dx=np.float32(1.0)
dt=np.float32(0.005)
inv2dx=np.float32(0.5/dx)
mass=np.float32(1.0)
thermal_scale=np.float32(1.0)
alpha_par=np.float32(0.5)
alpha_perp=np.float32(0.5)
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

M_target=cp.zeros((Ncells,12),dtype=cp.float32)
lambda_io=cp.zeros((Ncells,12),dtype=cp.float32)
n_in=cp.ones(Ncells,dtype=cp.float32)
n_out=cp.zeros(Ncells,dtype=cp.float32)
upar=cp.zeros(Ncells,dtype=cp.float32)
Ppar=cp.zeros(Ncells,dtype=cp.float32)
Pperp=cp.zeros(Ncells,dtype=cp.float32)
qpar=cp.zeros(Ncells,dtype=cp.float32)
qperp=cp.zeros(Ncells,dtype=cp.float32)
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
    gyrotropic_12_kernel((blocks_g,),(threads_g,),
        (M_target,n_in,lambda_io,n_out,upar,Ppar,Pperp,qpar,qperp,
         status,np.int32(Ncells),mass,thermal_scale,alpha_par,alpha_perp))
    ct_emf_kernel(blocks,threads,
        (rho,mx,my,mz,Bx,By,Bz,Emfx,Emfy,Emfz,np.int32(Ni)))
    ct_curl_kernel(blocks,threads,
        (Emfx,Emfy,Emfz,Bx,By,Bz,np.int32(Ni),dt,inv2dx))
    glm_kernel(blocks,threads,
        (Bx,By,Bz,psi,np.int32(Ni),dt,dx,np.float32(1.0),np.float32(0.5)))
    if step % 5 == 0:
        print(f"step {step:3d}  status={cp.unique(status)}  "
              f"mean|B|={float(cp.mean(cp.sqrt(Bx**2+By**2+Bz**2))):.4f}")

print("Done.")
print("Final mean Ppar / Pperp =", float(cp.mean(Ppar)), float(cp.mean(Pperp)))
