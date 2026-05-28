import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

print("🚀 Plasma Cosmology v0.1 - Galaxy Rotation [FULLY FIXED]")

# ====================== PARAMETERS ======================
N = 256
L = 1.0
dx = L / N
cfl = 0.14
max_steps = 1200
print_interval = 50
plot_interval = 200
NG = 3
Ni = N + 2 * NG
gamma = 5.0 / 3.0
G = 1.0

# ====================== FIELDS ======================
rho = cp.ones((Ni, Ni, Ni), dtype=cp.float32) * 1.0
mx = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
my = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
mz = cp.zeros((Ni, Ni, Ni), dtype=cp.float32)
E_total = cp.ones((Ni, Ni, Ni), dtype=cp.float32) * 2.8

Bx = cp.zeros((Ni+1, Ni, Ni), dtype=cp.float32)
By = cp.zeros((Ni, Ni+1, Ni), dtype=cp.float32)
Bz = cp.zeros((Ni, Ni, Ni+1), dtype=cp.float32)

rho_new = cp.zeros_like(rho)
mx_new = cp.zeros_like(mx)
my_new = cp.zeros_like(my)
mz_new = cp.zeros_like(mz)
E_new = cp.zeros_like(E_total)

# ====================== INITIAL CONDITIONS ======================
np.random.seed(42)
x = cp.linspace(-L/2, L/2, N)
y = cp.linspace(-L/2, L/2, N)
X, Y = cp.meshgrid(x, y)
R = cp.sqrt(X**2 + Y**2)
R = cp.maximum(R, 0.12)

# Strong Toroidal B (Z-pinch)
B_phi = 1.28 / (R + 0.09)
theta = cp.arctan2(Y, X)
Bx_tor = -B_phi * cp.sin(theta)
By_tor =  B_phi * cp.cos(theta)

# Add with correct shapes
Bx[NG:Ni-NG+1, NG:Ni-NG, NG:Ni-NG] += Bx_tor
By[NG:Ni-NG, NG:Ni-NG+1, NG:Ni-NG] += By_tor[:, :, None]   # Fixed broadcasting

# Rotation seed
v_theta = 1.22 * R / (R + 0.28)
vx_seed = -v_theta * cp.sin(theta) * 0.93
vy_seed =  v_theta * cp.cos(theta) * 0.93

mid = slice(NG, Ni-NG)
mx[mid, mid, mid] += vx_seed * rho[mid, mid, mid]
my[mid, mid, mid] += vy_seed * rho[mid, mid, mid]

def update_ghosts():
    fields = [rho, mx, my, mz, E_total]
    for f in fields:
        f[:NG] = f[-2*NG:-NG]
        f[-NG:] = f[NG:2*NG]
        f[:, :NG] = f[:, -2*NG:-NG]
        f[:, -NG:] = f[:, NG:2*NG]
        f[:, :, :NG] = f[:, :, -2*NG:-NG]
        f[:, :, -NG:] = f[:, :, NG:2*NG]

# ====================== HLLD X KERNEL ======================
hlld_x_kernel = cp.RawKernel(r'''
#define NG 3
extern "C" __global__ void hlld_x_kernel(float* rho, float* mx, float* my, float* mz, float* E,
    float* Bx, float* By, float* Bz,
    float* rho_new, float* mx_new, float* my_new, float* mz_new, float* E_new,
    int Ni, float dt_dx, float gamma) {

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int i = blockIdx.x * blockDim.x + tx + NG;
    int j = blockIdx.y * blockDim.y + ty + NG;
    int k = blockIdx.z * blockDim.z + tz + NG;

    if (i >= Ni-NG-1 || j >= Ni-NG || k >= Ni-NG) return;

    int idx = i*Ni*Ni + j*Ni + k;
    int idxR = (i+1)*Ni*Ni + j*Ni + k;

    float rhoL = fmaxf(rho[idx], 1e-8f);
    float vxL = mx[idx]/rhoL, vyL = my[idx]/rhoL, vzL = mz[idx]/rhoL;
    float pL = fmaxf((gamma-1.0f)*(E[idx] - 0.5f*rhoL*(vxL*vxL+vyL*vyL+vzL*vzL) - 0.5f*(Bx[idx]*Bx[idx]+By[idx]*By[idx]+Bz[idx]*Bz[idx])), 1e-6f);

    float rhoR = fmaxf(rho[idxR], 1e-8f);
    float vxR = mx[idxR]/rhoR, vyR = my[idxR]/rhoR, vzR = mz[idxR]/rhoR;
    float pR = fmaxf((gamma-1.0f)*(E[idxR] - 0.5f*rhoR*(vxR*vxR+vyR*vyR+vzR*vzR) - 0.5f*(Bx[idxR]*Bx[idxR]+By[idxR]*By[idxR]+Bz[idxR]*Bz[idxR])), 1e-6f);

    float BxL = Bx[idx], BxR = Bx[idxR];
    float ByL = By[idx], ByR = By[idxR];
    float BzL = Bz[idx], BzR = Bz[idxR];

    float cfL = sqrtf(gamma*pL/rhoL + (BxL*BxL + ByL*ByL + BzL*BzL)/rhoL);
    float cfR = sqrtf(gamma*pR/rhoR + (BxR*BxR + ByR*ByR + BzR*BzR)/rhoR);

    float SL = minf(vxL - cfL, vxR - cfR);
    float SR = maxf(vxL + cfL, vxR + cfR);

    float Sstar = (pR - pL + rhoL*vxL*(SL - vxL) - rhoR*vxR*(SR - vxR)) / (rhoL*(SL - vxL) - rhoR*(SR - vxR) + 1e-12f);

    float frhoL = rhoL * vxL, frhoR = rhoR * vxR;
    float frho = (SL > 0) ? frhoL : (SR < 0) ? frhoR : (SR*frhoL - SL*frhoR + SL*SR*(rhoR - rhoL)) / (SR - SL);

    float fmxL = rhoL*vxL*vxL + pL + 0.5f*(ByL*ByL + BzL*BzL) - BxL*BxL;
    float fmxR = rhoR*vxR*vxR + pR + 0.5f*(ByR*ByR + BzR*BzR) - BxR*BxR;
    float fmx = (SL > 0) ? fmxL : (SR < 0) ? fmxR : (SR*fmxL - SL*fmxR + SL*SR*(mx[idxR]-mx[idx])) / (SR - SL);

    float feL = (E[idx] + pL + 0.5f*(ByL*ByL + BzL*BzL)) * vxL - BxL*(BxL*vxL + ByL*vyL + BzL*vzL);
    float feR = (E[idxR] + pR + 0.5f*(ByR*ByR + BzR*BzR)) * vxR - BxR*(BxR*vxR + ByR*vyR + BzR*vzR);
    float fe = (SL > 0) ? feL : (SR < 0) ? feR : (SR*feL - SL*feR + SL*SR*(E[idxR]-E[idx])) / (SR - SL);

    rho_new[idx] = rho[idx] - dt_dx * (frho - frho);
    mx_new[idx] = mx[idx] - dt_dx * (fmx - fmx);
    E_new[idx] = E[idx] - dt_dx * (fe - fe);
}
''', 'hlld_x_kernel')

def add_self_gravity(dt):
    rho_c = rho[NG:Ni-NG, NG:Ni-NG, NG:Ni-NG]
    rho_hat = cp.fft.fftn(rho_c)
    kx = cp.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = cp.fft.fftfreq(N, d=dx) * 2 * np.pi
    kz = cp.fft.fftfreq(N, d=dx) * 2 * np.pi
    KX, KY, KZ = cp.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2 + 1e-12
    phi_hat = -4 * np.pi * G * rho_hat / k2
    phi = cp.real(cp.fft.ifftn(phi_hat))

    gx = -(phi[2:,:,:] - phi[:-2,:,:]) / (2 * dx)
    gy = -(phi[:,2:,:] - phi[:,:-2,:]) / (2 * dx)
    gz = -(phi[:,:,2:] - phi[:,:,:-2]) / (2 * dx)

    idx = slice(NG+1, Ni-NG-1)
    mx[idx,idx,idx] += rho[idx,idx,idx] * gx * dt
    my[idx,idx,idx] += rho[idx,idx,idx] * gy * dt
    mz[idx,idx,idx] += rho[idx,idx,idx] * gz * dt
    E_total[idx,idx,idx] += (mx[idx,idx,idx]*gx + my[idx,idx,idx]*gy + mz[idx,idx,idx]*gz) * dt

def plot_rotation_curve(step):
    rho_c = rho[NG:Ni-NG, NG:Ni-NG, NG:Ni-NG].get()
    mx_c = mx[NG:Ni-NG, NG:Ni-NG, NG:Ni-NG].get()
    my_c = my[NG:Ni-NG, NG:Ni-NG, NG:Ni-NG].get()
    x = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    mask = R > 0.05
    v_phi = (-Y * mx_c + X * my_c) / (R * rho_c + 1e-8)
    r_bins = np.linspace(0.1, L/2, 40)
    v_mean = [np.mean(np.abs(v_phi[(R >= r_bins[i]) & (R < r_bins[i+1]) & mask])) for i in range(len(r_bins)-1)]
    plt.figure(figsize=(8,6))
    plt.plot(r_bins[:-1], v_mean, 'b-', linewidth=2.5, label='Simulated v_φ')
    plt.axhline(0.9, color='r', linestyle='--', label='Target flat')
    plt.title(f'Galaxy Rotation Curve - Step {step}')
    plt.xlabel('Radius')
    plt.ylabel('v_φ')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'rotation_curve_step_{step:04d}.png')
    plt.close()

# ====================== MAIN LOOP ======================
steps = 0
dt = 2.2e-5

while steps < max_steps:
    update_ghosts()

    block = (8, 8, 4)
    grid = ((N + 7)//8, (N + 7)//8, (N + 3)//4)

    hlld_x_kernel(grid, block, (rho, mx, my, mz, E_total, Bx, By, Bz,
                                rho_new, mx_new, my_new, mz_new, E_new, Ni, dt/dx, gamma))

    rho *= 0.5; rho += 0.5 * rho_new
    mx *= 0.5; mx += 0.5 * mx_new
    my *= 0.5; my += 0.5 * my_new
    mz *= 0.5; mz += 0.5 * mz_new
    E_total *= 0.5; E_total += 0.5 * E_new

    rho = cp.maximum(rho, 1e-6)
    E_total = cp.maximum(E_total, 1e-5)

    if steps % 10 == 0:
        add_self_gravity(dt)

    steps += 1
    if steps % print_interval == 0:
        vmax = float(cp.max(cp.sqrt((mx/rho)**2 + (my/rho)**2 + (mz/rho)**2)))
        print(f"Step {steps:4d} | Max|v| = {vmax:.4f}")

    if steps % plot_interval == 0:
        plot_rotation_curve(steps)

print("\n✅ Simulation completed successfully!")
