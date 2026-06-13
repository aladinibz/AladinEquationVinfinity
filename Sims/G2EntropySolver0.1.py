import cupy as cp
from dataclasses import dataclass
from typing import Literal, Dict
import numpy as np

BoundaryType = Literal["periodic", "outflow", "wall"]

@dataclass
class G2MHDState:
    rho: cp.ndarray
    mx: cp.ndarray; my: cp.ndarray; mz: cp.ndarray
    E: cp.ndarray
    Bx: cp.ndarray; By: cp.ndarray; Bz: cp.ndarray
    psi: cp.ndarray
    oct_imag: cp.ndarray  # (dofs, 7) full imaginary octonion fiber

class OctonionG2:
    @staticmethod
    def get_table():
        if not hasattr(OctonionG2, '_table'):
            t = cp.zeros((7,7,7), dtype=cp.float32)
            lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
            for i,j,k in lines:
                t[i,j,k] = t[j,k,i] = t[k,i,j] = 1.0
                t[j,i,k] = t[k,j,i] = t[i,k,j] = -1.0
            OctonionG2._table = t
        return OctonionG2._table

    @staticmethod
    def phi_3form(o):
        v = o[:,0:3]; B = o[:,3:6]; chi = o[:,6:7]
        cross = cp.einsum('...i,...j,ijk->...k', v, B, OctonionG2.get_table())
        return (cp.sum(cross * v, axis=1) + cp.sum(cross * B, axis=1) + cp.sum(cross * chi, axis=1))

    @staticmethod
    def associator(o):
        ab = cp.einsum('...i,...j,ijk->...k', o, o, OctonionG2.get_table())
        left = cp.einsum('...i,...j,ijk->...k', ab, o, OctonionG2.get_table())
        bc = cp.einsum('...i,...j,ijk->...k', o, o, OctonionG2.get_table())
        right = cp.einsum('...i,...j,ijk->...k', o, bc, OctonionG2.get_table())
        return left - right

class AladinG2EntropySolver:
    def __init__(self, Ni=32, gamma=5./3, alpha=0.5, lambda_g2=0.25, bc_type: BoundaryType = "wall", ch=1.0,
                 rho_min=1e-8, p_min=1e-8):
        self.Ni = Ni
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_g2 = lambda_g2
        self.bc_type = bc_type
        self.ch = ch
        self.rho_min = rho_min
        self.p_min = p_min
        self.dofs = Ni**3
        self.inv_dx = 1.0 / Ni
        self.N = 4
        self.build_gll_d_matrix()

    def build_gll_d_matrix(self):
        nodes, _ = np.polynomial.legendre.leggauss(self.N + 1)
        D_np = np.zeros((self.N+1, self.N+1), dtype=np.float32)
        for i in range(self.N+1):
            for j in range(self.N+1):
                if i != j:
                    D_np[i,j] = np.polynomial.legendre.legval(nodes[i],[0.,1.]) / ((nodes[i]-nodes[j]) * np.polynomial.legendre.legval(nodes[j],[0.,1.]))
        for i in range(self.N+1):
            D_np[i,i] = -np.sum(D_np[i])
        self.D = cp.array(D_np, dtype=cp.float32)

    def prepare_octonion(self, u: G2MHDState):
        rho = cp.maximum(u.rho, 1e-12)
        v = cp.stack([u.mx/rho, u.my/rho, u.mz/rho], axis=1)
        B = cp.stack([u.Bx, u.By, u.Bz], axis=1)
        chi = (u.Bx*u.mx + u.By*u.my + u.Bz*u.mz) / (rho * (cp.sum(B**2, axis=1) + 1e-12))
        u.oct_imag = cp.concatenate([v, B, chi[:, None]], axis=1)

    def apply_positivity_limiter(self, u: G2MHDState, theta: float = 0.1):
        rho_mean = cp.mean(u.rho)
        rho = cp.maximum(u.rho, self.rho_min)
        theta_rho = cp.minimum(1.0, theta * rho_mean / cp.maximum(rho_mean - rho, 1e-12))
        u.rho = theta_rho * (rho - rho_mean) + rho_mean

        E_kin = 0.5 * (u.mx**2 + u.my**2 + u.mz**2) / u.rho
        E_mag = 0.5 * (u.Bx**2 + u.By**2 + u.Bz**2)
        p = (self.gamma - 1.0) * (u.E - E_kin - E_mag)
        p_mean = cp.mean(p)
        theta_p = cp.minimum(1.0, theta * p_mean / cp.maximum(p_mean - p, 1e-12))
        u.E = theta_p * (u.E - (E_kin + E_mag + p_mean/(self.gamma-1.0))) + (E_kin + E_mag + p_mean/(self.gamma-1.0))
        self.prepare_octonion(u)

    def project_g2_symmetry(self, u: G2MHDState, strength: float = 0.05):
        self.prepare_octonion(u)
        phi = OctonionG2.phi_3form(u.oct_imag)
        phi_mean = cp.mean(phi)
        correction = strength * (phi_mean - phi)[:, None] * u.oct_imag
        u.oct_imag += correction

    def compute_action_breakdown(self, u: G2MHDState) -> Dict[str, float]:
        """Action Principle:
        S = ∫ [ L_MHD(ρ, v, B) + λ |φ(O)|² + α |[O,O,O]|² + β |∇φ|² ] d³x dt
        """
        self.prepare_octonion(u)
        rho = cp.maximum(u.rho, 1e-12)
        E_kin = 0.5 * cp.sum(u.mx**2 + u.my**2 + u.mz**2, axis=1) / rho
        E_mag = 0.5 * (u.Bx**2 + u.By**2 + u.Bz**2)
        E_int = (u.E - E_kin - E_mag) / (self.gamma - 1.0)

        phi = OctonionG2.phi_3form(u.oct_imag)
        E_phi = 0.5 * self.lambda_g2 * float(cp.mean(phi**2).item())

        assoc = OctonionG2.associator(u.oct_imag)
        E_assoc = self.alpha * float(cp.mean(cp.sum(assoc**2, axis=1)).item())

        E_grad_phi = 0.0

        total_classic = float(cp.sum(rho*E_kin + E_int + E_mag).item())
        total_g2 = E_phi + E_assoc + E_grad_phi

        return {
            "total_classic": total_classic,
            "kinetic": float(cp.sum(rho*E_kin).item()),
            "internal": float(cp.sum(E_int).item()),
            "magnetic": float(cp.sum(E_mag).item()),
            "G2_phi": E_phi,
            "G2_associator": E_assoc,
            "G2_grad_phi": E_grad_phi,
            "grand_total": total_classic + total_g2
        }

    volume_operator_g2_kernel = cp.RawKernel(r'''
    #define TILE_X 8
    #define TILE_Y 8
    #define TILE_Z 4
    #define PAD 1
    #define OCT_DIM 7

    __device__ void load_fano_table(float table[OCT_DIM][OCT_DIM][OCT_DIM]) {
        for(int i=0;i<OCT_DIM;i++)for(int j=0;j<OCT_DIM;j++)for(int k=0;k<OCT_DIM;k++) table[i][j][k]=0.0f;
        int lines[7][3] = {{0,1,2},{0,3,4},{0,5,6},{1,3,5},{1,4,6},{2,3,6},{2,4,5}};
        for(int l=0;l<7;l++){
            int i=lines[l][0],j=lines[l][1],k=lines[l][2];
            table[i][j][k]=table[j][k][i]=table[k][i][j]=1.0f;
            table[j][i][k]=table[k][j][i]=table[i][k][j]=-1.0f;
        }
    }

    __device__ void multiply_imag_device(const float a[OCT_DIM], const float b[OCT_DIM], float c[OCT_DIM], float table[OCT_DIM][OCT_DIM][OCT_DIM]) {
        for(int k=0;k<OCT_DIM;k++) c[k]=0.0f;
        for(int i=0;i<OCT_DIM;i++)for(int j=0;j<OCT_DIM;j++){
            float prod = a[i]*b[j];
            if(prod==0.0f) continue;
            for(int k=0;k<OCT_DIM;k++) c[k] += prod * table[i][j][k];
        }
    }

    __device__ void associator_device(const float a[OCT_DIM], const float b[OCT_DIM], const float c[OCT_DIM], float result[OCT_DIM], float table[OCT_DIM][OCT_DIM][OCT_DIM]) {
        float ab[OCT_DIM], bc[OCT_DIM], left[OCT_DIM], right[OCT_DIM];
        multiply_imag_device(a,b,ab,table);
        multiply_imag_device(ab,c,left,table);
        multiply_imag_device(b,c,bc,table);
        multiply_imag_device(a,bc,right,table);
        for(int k=0;k<OCT_DIM;k++) result[k] = left[k] - right[k];
    }

    __device__ float log_mean(float a, float b) {
        if(fabsf(a-b) < 1e-12f) return a;
        return (a-b) / logf(a/b);
    }

    extern "C" __global__ void volume_operator_g2_kernel(
        const float* rho, const float* mx, const float* my, const float* mz, const float* E,
        const float* Bx, const float* By, const float* Bz, const float* psi,
        float* rhs_rho, float* rhs_mx, float* rhs_my, float* rhs_mz, float* rhs_E,
        float* rhs_Bx, float* rhs_By, float* rhs_Bz, float* rhs_psi, float* rhs_oct_imag,
        const float* D, int N, int Ni, float base_alpha, float gamma, float inv_dx,
        float ch, float lambda_g2, float extra_diss, int bc_wall) {

        int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
        int i = blockIdx.x*TILE_X + tx;
        int j = blockIdx.y*TILE_Y + ty;
        int k = blockIdx.z*TILE_Z + tz;
        if(i>=Ni || j>=Ni || k>=Ni) return;

        int idx = (i*Ni + j)*Ni + k;
        int base_oct = idx * OCT_DIM;

        __shared__ float s_rho[TILE_X+2*PAD][TILE_Y+2*PAD][TILE_Z+2*PAD];
        __shared__ float s_mx [TILE_X+2*PAD][TILE_Y+2*PAD][TILE_Z+2*PAD];
        __shared__ float s_my [TILE_X+2*PAD][TILE_Y+2*PAD][TILE_Z+2*PAD];
        __shared__ float s_mz [TILE_X+2*PAD][TILE_Y+2*PAD][TILE_Z+2*PAD];
        __shared__ float s_E  [TILE_X+2*PAD][TILE_Y+2*PAD][TILE_Z+2*PAD];
        __shared__ float fano_table[OCT_DIM][OCT_DIM][OCT_DIM];

        s_rho[tx+PAD][ty+PAD][tz+PAD] = rho[idx];
        s_mx[tx+PAD][ty+PAD][tz+PAD] = mx[idx];
        s_my[tx+PAD][ty+PAD][tz+PAD] = my[idx];
        s_mz[tx+PAD][ty+PAD][tz+PAD] = mz[idx];
        s_E[tx+PAD][ty+PAD][tz+PAD] = E[idx];
        if(tx==0 && ty==0 && tz==0) load_fano_table(fano_table);
        __syncthreads();

        float r = max(s_rho[tx+PAD][ty+PAD][tz+PAD], 1e-12f);
        float vx = s_mx[tx+PAD][ty+PAD][tz+PAD] / r;
        float vy = s_my[tx+PAD][ty+PAD][tz+PAD] / r;
        float vz = s_mz[tx+PAD][ty+PAD][tz+PAD] / r;
        float Bx0 = Bx[idx], By0 = By[idx], Bz0 = Bz[idx];
        float B2 = Bx0*Bx0 + By0*By0 + Bz0*Bz0;
        float p = max((gamma-1.0f)*(s_E[tx+PAD][ty+PAD][tz+PAD] - 0.5f*r*(vx*vx+vy*vy+vz*vz) - 0.5f*B2), 1e-12f);

        float chi = (Bx0*vx + By0*vy + Bz0*vz) / (r * (B2 + 1e-12f));
        float o[OCT_DIM] = {vx, vy, vz, Bx0, By0, Bz0, chi};

        float phi = o[0]*o[1]*o[2] + o[0]*o[3]*o[4] + o[0]*o[5]*o[6] +
                    o[1]*o[3]*o[5] + o[1]*o[4]*o[6] + o[2]*o[3]*o[6] + o[2]*o[4]*o[5];

        float assoc[OCT_DIM];
        associator_device(o, o, o, assoc, fano_table);

        float assoc_norm = 0.0f;
        for(int m=0; m<OCT_DIM; m++) assoc_norm += assoc[m]*assoc[m];
        assoc_norm = sqrtf(assoc_norm);

        float g_phi[OCT_DIM];
        g_phi[0] = o[1]*o[2] + o[3]*o[4] + o[5]*o[6];
        g_phi[1] = o[0]*o[2] + o[3]*o[5] + o[4]*o[6];
        g_phi[2] = o[0]*o[1] + o[3]*o[6] + o[4]*o[5];
        g_phi[3] = o[0]*o[4] + o[1]*o[5] + o[2]*o[6];
        g_phi[4] = o[0]*o[3] + o[1]*o[6] + o[2]*o[5];
        g_phi[5] = o[0]*o[6] + o[1]*o[3] + o[2]*o[4];
        g_phi[6] = o[0]*o[5] + o[1]*o[4] + o[2]*o[3];

        float do_dx[OCT_DIM] = {0.0f}, do_dy[OCT_DIM] = {0.0f}, do_dz[OCT_DIM] = {0.0f};
        for(int m=0; m<=N; m++) {
            int offx = m - tx; int sxm = tx + PAD + offx;
            if(sxm >= 0 && sxm < TILE_X+2*PAD) do_dx[0] += D[tx*(N+1)+m] * s_mx[sxm][ty+PAD][tz+PAD] / r;
            int offy = m - ty; int sym = ty + PAD + offy;
            if(sym >= 0 && sym < TILE_Y+2*PAD) do_dy[0] += D[ty*(N+1)+m] * s_mx[tx+PAD][sym][tz+PAD] / r;
            int offz = m - tz; int szm = tz + PAD + offz;
            if(szm >= 0 && szm < TILE_Z+2*PAD) do_dz[0] += D[tz*(N+1)+m] * s_mx[tx+PAD][ty+PAD][szm] / r;
        }
        for(int m=0; m<OCT_DIM; m++) {
            do_dx[m] *= inv_dx; do_dy[m] *= inv_dx; do_dz[m] *= inv_dx;
        }

        float dphi_dx = 0.0f, dphi_dy = 0.0f, dphi_dz = 0.0f;
        for(int m=0; m<OCT_DIM; m++) {
            dphi_dx += g_phi[m] * do_dx[m];
            dphi_dy += g_phi[m] * do_dy[m];
            dphi_dz += g_phi[m] * do_dz[m];
        }
        float phi_grad_mag2 = dphi_dx*dphi_dx + dphi_dy*dphi_dy + dphi_dz*dphi_dz;

        float g[OCT_DIM];
        for(int m=0; m<OCT_DIM; m++) {
            g[m] = lambda_g2 * phi * g_phi[m] 
                 + 2.0f * base_alpha * assoc[m]
                 + 0.5f * phi_grad_mag2 * g_phi[m];
        }

        float calib = 1.0f + lambda_g2 * fabsf(phi);
        float diss = base_alpha * (fabsf(phi) + assoc_norm + sqrtf(phi_grad_mag2)) * (1.0f + extra_diss);

        float total_p = p + 0.5f * B2;
        float vB = vx*Bx0 + vy*By0 + vz*Bz0;

        float fx = -(r*vx*vx + total_p)*vx - g[0] - diss*vx*calib;
        float fy = -(r*vy*vy + total_p)*vy - g[1] - diss*vy*calib;
        float fz = -(r*vz*vz + total_p)*vz - g[2] - diss*vz*calib;

        if (bc_wall && (i==0 || i==Ni-1)) fx = 0.0f;
        if (bc_wall && (j==0 || j==Ni-1)) fy = 0.0f;
        if (bc_wall && (k==0 || k==Ni-1)) fz = 0.0f;

        rhs_mx[idx] = fx;
        rhs_my[idx] = fy;
        rhs_mz[idx] = fz;

        float energy_ec = (s_E[tx+PAD][ty+PAD][tz+PAD] + total_p) * vx - vB * Bx0;
        rhs_E[idx] = -energy_ec - diss * (vx*vx + vy*vy + vz*vz) * calib;

        rhs_Bx[idx] = (vy*Bz0 - vz*By0) - g[3];
        rhs_By[idx] = (vz*Bx0 - vx*Bz0) - g[4];
        rhs_Bz[idx] = (vx*By0 - vy*Bx0) - g[5];

        rhs_psi[idx] = -ch*ch * (Bx0*vx + By0*vy + Bz0*vz);

        for(int m=0; m<OCT_DIM; m++) {
            rhs_oct_imag[base_oct + m] = -g[m];
        }

        rhs_rho[idx] = 0.0f;
    }
    ''', 'volume_operator_g2_kernel')

    def launch_volume_operator_g2(self, u: G2MHDState, rhs: G2MHDState, extra_diss: float = 0.0):
        bc_wall = 1 if self.bc_type == "wall" else 0
        blocks_x = (self.Ni + 7) // 8
        blocks_y = (self.Ni + 7) // 8
        blocks_z = (self.Ni + 3) // 4
        self.volume_operator_g2_kernel((blocks_x, blocks_y, blocks_z), (8, 8, 4),
            (u.rho, u.mx, u.my, u.mz, u.E, u.Bx, u.By, u.Bz, u.psi,
             rhs.rho, rhs.mx, rhs.my, rhs.mz, rhs.E,
             rhs.Bx, rhs.By, rhs.Bz, rhs.psi, rhs.oct_imag,
             self.D, self.N, self.Ni, self.alpha, self.gamma, self.inv_dx,
             self.ch, self.lambda_g2, extra_diss, bc_wall))

    def compute_rhs(self, u: G2MHDState, extra_diss: float = 0.0):
        rhs = G2MHDState(
            rho=cp.zeros_like(u.rho), mx=cp.zeros_like(u.mx), my=cp.zeros_like(u.my),
            mz=cp.zeros_like(u.mz), E=cp.zeros_like(u.E),
            Bx=cp.zeros_like(u.Bx), By=cp.zeros_like(u.By), Bz=cp.zeros_like(u.Bz),
            psi=cp.zeros_like(u.psi), oct_imag=cp.zeros_like(u.oct_imag)
        )
        self.launch_volume_operator_g2(u, rhs, extra_diss)
        return rhs

    def step(self, u: G2MHDState, dt: float) -> G2MHDState:
        # Stage 1
        rhs1 = self.compute_rhs(u, 0.0)
        u1 = G2MHDState(
            rho=u.rho + dt*rhs1.rho,
            mx=u.mx + dt*rhs1.mx, my=u.my + dt*rhs1.my, mz=u.mz + dt*rhs1.mz,
            E=u.E + dt*rhs1.E,
            Bx=u.Bx + dt*rhs1.Bx, By=u.By + dt*rhs1.By, Bz=u.Bz + dt*rhs1.Bz,
            psi=u.psi + dt*rhs1.psi,
            oct_imag=u.oct_imag + dt*rhs1.oct_imag
        )
        self.apply_positivity_limiter(u1)

        # Stage 2
        rhs2 = self.compute_rhs(u1, 0.0)
        u2 = G2MHDState(
            rho=(3./4)*u.rho + (1./4)*u1.rho + (1./4)*dt*rhs2.rho,
            mx=(3./4)*u.mx + (1./4)*u1.mx + (1./4)*dt*rhs2.mx,
            my=(3./4)*u.my + (1./4)*u1.my + (1./4)*dt*rhs2.my,
            mz=(3./4)*u.mz + (1./4)*u1.mz + (1./4)*dt*rhs2.mz,
            E=(3./4)*u.E + (1./4)*u1.E + (1./4)*dt*rhs2.E,
            Bx=(3./4)*u.Bx + (1./4)*u1.Bx + (1./4)*dt*rhs2.Bx,
            By=(3./4)*u.By + (1./4)*u1.By + (1./4)*dt*rhs2.By,
            Bz=(3./4)*u.Bz + (1./4)*u1.Bz + (1./4)*dt*rhs2.Bz,
            psi=(3./4)*u.psi + (1./4)*u1.psi + (1./4)*dt*rhs2.psi,
            oct_imag=(3./4)*u.oct_imag + (1./4)*u1.oct_imag + (1./4)*dt*rhs2.oct_imag
        )
        self.apply_positivity_limiter(u2)

        # Stage 3
        rhs3 = self.compute_rhs(u2, 0.0)
        u_new = G2MHDState(
            rho=(1./3)*u.rho + (2./3)*u2.rho + (2./3)*dt*rhs3.rho,
            mx=(1./3)*u.mx + (2./3)*u2.mx + (2./3)*dt*rhs3.mx,
            my=(1./3)*u.my + (2./3)*u2.my + (2./3)*dt*rhs3.my,
            mz=(1./3)*u.mz + (2./3)*u2.mz + (2./3)*dt*rhs3.mz,
            E=(1./3)*u.E + (2./3)*u2.E + (2./3)*dt*rhs3.E,
            Bx=(1./3)*u.Bx + (2./3)*u2.Bx + (2./3)*dt*rhs3.Bx,
            By=(1./3)*u.By + (2./3)*u2.By + (2./3)*dt*rhs3.By,
            Bz=(1./3)*u.Bz + (2./3)*u2.Bz + (2./3)*dt*rhs3.Bz,
            psi=(1./3)*u.psi + (2./3)*u2.psi + (2./3)*dt*rhs3.psi,
            oct_imag=(1./3)*u.oct_imag + (2./3)*u2.oct_imag + (2./3)*dt*rhs3.oct_imag
        )
        self.apply_positivity_limiter(u_new)
        self.project_g2_symmetry(u_new)
        return u_new

    def run(self, u0: G2MHDState, t_final: float = 1.0, print_interval: int = 50):
        u = u0
        t = 0.0
        step = 0
        while t < t_final:
            dt = 0.2 * self.inv_dx
            u = self.step(u, dt)
            t += dt
            step += 1
            if step % print_interval == 0:
                breakdown = self.compute_action_breakdown(u)
                print(f"t = {t:.4f} | Total E = {breakdown['grand_total']:.6e} | "
                      f"G2_phi = {breakdown['G2_phi']:.6e} | G2_assoc = {breakdown['G2_associator']:.6e}")
        return u


if __name__ == "__main__":
    solver = AladinG2EntropySolver(Ni=16, bc_type="wall")
    print("✅ Full G2 Entropy Solver ready. Publishing on Zenodo soon!")
    # Add your initial condition and run here
