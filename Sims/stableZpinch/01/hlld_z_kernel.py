"""
ALADIN Plasma Drive v2.1
HLLD-Z RawKernel — Explicit Cylindrical

Colab Pro A100. Pure CuPy RawKernel.
Miyoshi–Kusano 2005 on face_z.
Normal = z, transverse = (r, φ).
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

hlld_z_kernel = None
if HAS_CUPY:
    hlld_z_kernel = cp.RawKernel(r"""
#define GAMMA 1.6666667f
#define GAMMA_M1 0.6666667f
#define SMALL 1e-12f
#define SMALL_P 1e-10f

__device__ __forceinline__ void prim_from_cons(
    float rho, float m_n, float m_t1, float m_t2, float E,
    float Bn, float Bt1, float Bt2,
    float& vn, float& vt1, float& vt2, float& p, float& ptot
) {
    float rho_s = fmaxf(rho, SMALL);
    vn  = m_n  / rho_s;
    vt1 = m_t1 / rho_s;
    vt2 = m_t2 / rho_s;
    float ke = 0.5f * rho_s * (vn*vn + vt1*vt1 + vt2*vt2);
    float be = 0.5f * (Bn*Bn + Bt1*Bt1 + Bt2*Bt2);
    p = fmaxf(GAMMA_M1 * (E - ke - be), SMALL_P);
    ptot = p + be;
}

__device__ void hlld_flux_local(
    float rL, float vnL, float vt1L, float vt2L, float pL,
    float Bt1L, float Bt2L,
    float rR, float vnR, float vt1R, float vt2R, float pR,
    float Bt1R, float Bt2R,
    float Bn,
    float& f_rho, float& f_mn, float& f_mt1, float& f_mt2,
    float& f_E, float& f_Bt1, float& f_Bt2
) {
    float ptL = pL + 0.5f * (Bn*Bn + Bt1L*Bt1L + Bt2L*Bt2L);
    float ptR = pR + 0.5f * (Bn*Bn + Bt1R*Bt1R + Bt2R*Bt2R);

    float cs2L = GAMMA * fmaxf(pL, SMALL_P) / fmaxf(rL, SMALL);
    float ca2L = (Bn*Bn + Bt1L*Bt1L + Bt2L*Bt2L) / fmaxf(rL, SMALL);
    float can2L = Bn*Bn / fmaxf(rL, SMALL);
    float discL = fmaxf((cs2L + ca2L)*(cs2L + ca2L) - 4.f*cs2L*can2L, 0.f);
    float cfL = sqrtf(0.5f * (cs2L + ca2L + sqrtf(discL)));

    float cs2R = GAMMA * fmaxf(pR, SMALL_P) / fmaxf(rR, SMALL);
    float ca2R = (Bn*Bn + Bt1R*Bt1R + Bt2R*Bt2R) / fmaxf(rR, SMALL);
    float can2R = Bn*Bn / fmaxf(rR, SMALL);
    float discR = fmaxf((cs2R + ca2R)*(cs2R + ca2R) - 4.f*cs2R*can2R, 0.f);
    float cfR = sqrtf(0.5f * (cs2R + ca2R + sqrtf(discR)));

    float SL = fminf(vnL - cfL, vnR - cfR);
    float SR = fmaxf(vnL + cfL, vnR + cfR);

    float den = (SR - vnR)*rR - (SL - vnL)*rL;
    float S_star = ((SR - vnR)*rR*vnR - (SL - vnL)*rL*vnL - ptR + ptL) / (den + SMALL);

    float rL_star = rL * (SL - vnL) / (SL - S_star + SMALL);
    float rR_star = rR * (SR - vnR) / (SR - S_star + SMALL);
    float pt_star = ptL + rL * (SL - vnL) * (S_star - vnL);

    float denomL = rL * (SL - vnL) * (SL - S_star) - Bn*Bn;
    float denomR = rR * (SR - vnR) * (SR - S_star) - Bn*Bn;

    float Bt1L_star = Bt1L * (rL * (SL - vnL)*(SL - vnL) - Bn*Bn) / (denomL + SMALL);
    float Bt2L_star = Bt2L * (rL * (SL - vnL)*(SL - vnL) - Bn*Bn) / (denomL + SMALL);
    float Bt1R_star = Bt1R * (rR * (SR - vnR)*(SR - vnR) - Bn*Bn) / (denomR + SMALL);
    float Bt2R_star = Bt2R * (rR * (SR - vnR)*(SR - vnR) - Bn*Bn) / (denomR + SMALL);

    float vt1L_star = vt1L - Bn * Bt1L * (S_star - vnL) / (denomL + SMALL);
    float vt2L_star = vt2L - Bn * Bt2L * (S_star - vnL) / (denomL + SMALL);
    float vt1R_star = vt1R - Bn * Bt1R * (S_star - vnR) / (denomR + SMALL);
    float vt2R_star = vt2R - Bn * Bt2R * (S_star - vnR) / (denomR + SMALL);

    float S_starL = S_star - fabsf(Bn) / sqrtf(fmaxf(rL_star, SMALL));
    float S_starR = S_star + fabsf(Bn) / sqrtf(fmaxf(rR_star, SMALL));

    float sqrt_rL = sqrtf(fmaxf(rL_star, SMALL));
    float sqrt_rR = sqrtf(fmaxf(rR_star, SMALL));
    float signBn = (Bn >= 0.f) ? 1.f : -1.f;

    float vt1_ss = (sqrt_rL * vt1L_star + sqrt_rR * vt1R_star + signBn * (Bt1R_star - Bt1L_star)) /
                   (sqrt_rL + sqrt_rR + SMALL);
    float vt2_ss = (sqrt_rL * vt2L_star + sqrt_rR * vt2R_star + signBn * (Bt2R_star - Bt2L_star)) /
                   (sqrt_rL + sqrt_rR + SMALL);
    float Bt1_ss = (sqrt_rL * Bt1R_star + sqrt_rR * Bt1L_star +
                    signBn * sqrt_rL * sqrt_rR * (vt1R_star - vt1L_star)) /
                   (sqrt_rL + sqrt_rR + SMALL);
    float Bt2_ss = (sqrt_rL * Bt2R_star + sqrt_rR * Bt2L_star +
                    signBn * sqrt_rL * sqrt_rR * (vt2R_star - vt2L_star)) /
                   (sqrt_rL + sqrt_rR + SMALL);

    float r, vn, vt1, vt2, Bt1, Bt2, pt;
    if (SL >= 0.f) {
        r = rL; vn = vnL; vt1 = vt1L; vt2 = vt2L; Bt1 = Bt1L; Bt2 = Bt2L; pt = ptL;
    } else if (S_starL >= 0.f) {
        r = rL_star; vn = S_star; vt1 = vt1L_star; vt2 = vt2L_star;
        Bt1 = Bt1L_star; Bt2 = Bt2L_star; pt = pt_star;
    } else if (S_star >= 0.f) {
        r = rL_star; vn = S_star; vt1 = vt1_ss; vt2 = vt2_ss;
        Bt1 = Bt1_ss; Bt2 = Bt2_ss; pt = pt_star;
    } else if (S_starR >= 0.f) {
        r = rR_star; vn = S_star; vt1 = vt1_ss; vt2 = vt2_ss;
        Bt1 = Bt1_ss; Bt2 = Bt2_ss; pt = pt_star;
    } else if (SR >= 0.f) {
        r = rR_star; vn = S_star; vt1 = vt1R_star; vt2 = vt2R_star;
        Bt1 = Bt1R_star; Bt2 = Bt2R_star; pt = pt_star;
    } else {
        r = rR; vn = vnR; vt1 = vt1R; vt2 = vt2R; Bt1 = Bt1R; Bt2 = Bt2R; pt = ptR;
    }

    float ke = 0.5f * r * (vn*vn + vt1*vt1 + vt2*vt2);
    float be = 0.5f * (Bn*Bn + Bt1*Bt1 + Bt2*Bt2);
    float p = fmaxf(pt - be, SMALL_P);
    float Etot = p / GAMMA_M1 + ke + be;

    f_rho = r * vn;
    f_mn  = r * vn * vn + pt - Bn * Bn;
    f_mt1 = r * vn * vt1 - Bn * Bt1;
    f_mt2 = r * vn * vt2 - Bn * Bt2;
    f_E   = (Etot + pt) * vn - Bn * (vn * Bn + vt1 * Bt1 + vt2 * Bt2);
    f_Bt1 = vn * Bt1 - vt1 * Bn;
    f_Bt2 = vn * Bt2 - vt2 * Bn;
}

extern "C" __global__
__launch_bounds__(512, 2)
void hlld_z_kernel(
    const float* __restrict__ rho,
    const float* __restrict__ rho_vr,
    const float* __restrict__ rho_vphi,
    const float* __restrict__ rho_vz,
    const float* __restrict__ E,
    const float* __restrict__ Br,
    const float* __restrict__ Bphi,
    const float* __restrict__ Bz,
    float* __restrict__ flux_z_rho,
    float* __restrict__ flux_z_vr,
    float* __restrict__ flux_z_vphi,
    float* __restrict__ flux_z_vz,
    float* __restrict__ flux_z_E,
    float* __restrict__ flux_z_Br,
    float* __restrict__ flux_z_Bphi,
    int nr, int nphi, int nz, int g
) {
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    int ip = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int nr_tot = nr + 2*g;
    int nphi_tot = nphi + 2*g;
    int nz_tot = nz + 2*g;

    if (iz < 1 || iz >= nz_tot || ir >= nr_tot || ip >= nphi_tot) return;

    int idxL = (ir * nphi_tot + ip) * nz_tot + (iz - 1);
    int idxR = (ir * nphi_tot + ip) * nz_tot + iz;
    int fidx = (ir * nphi_tot + ip) * nz_tot + iz;

    float rL = rho[idxL], mvrL = rho_vr[idxL], mvpL = rho_vphi[idxL], mvzL = rho_vz[idxL], EL = E[idxL];
    float rR = rho[idxR], mvrR = rho_vr[idxR], mvpR = rho_vphi[idxR], mvzR = rho_vz[idxR], ER = E[idxR];

    float Bn = Bz[fidx];
    float BrL = 0.5f * (Br[idxL] + Br[idxR]);
    float BrR = BrL;
    float BphiL = 0.5f * (Bphi[idxL] + Bphi[idxR]);
    float BphiR = BphiL;

    float vrL, vphiL, vzL, pL, ptL;
    float vrR, vphiR, vzR, pR, ptR;
    // normal = vz, t1 = vr, t2 = vphi
    prim_from_cons(rL, mvzL, mvrL, mvpL, EL, Bn, BrL, BphiL, vzL, vrL, vphiL, pL, ptL);
    prim_from_cons(rR, mvzR, mvrR, mvpR, ER, Bn, BrR, BphiR, vzR, vrR, vphiR, pR, ptR);

    float f_rho, f_mn, f_mt1, f_mt2, f_E, f_Bt1, f_Bt2;
    hlld_flux_local(rL, vzL, vrL, vphiL, pL, BrL, BphiL,
                    rR, vzR, vrR, vphiR, pR, BrR, BphiR,
                    Bn, f_rho, f_mn, f_mt1, f_mt2, f_E, f_Bt1, f_Bt2);

    flux_z_rho[fidx]  = f_rho;
    flux_z_vz[fidx]   = f_mn;
    flux_z_vr[fidx]   = f_mt1;
    flux_z_vphi[fidx] = f_mt2;
    flux_z_E[fidx]    = f_E;
    flux_z_Br[fidx]   = f_Bt1;
    flux_z_Bphi[fidx] = f_Bt2;
}
""", "hlld_z_kernel")


def launch_hlld_z(state, config, fluxes, block=(8, 8, 4)):
    """Launch HLLD-z. fluxes dict keys: rho, vr, vphi, vz, E, Br, Bphi."""
    if not HAS_CUPY or hlld_z_kernel is None:
        return
    nr, nphi, nz = config.nr, config.nphi, config.nz
    g = config.ghost_width
    grid = (
        (nr + 2*g + block[0] - 1) // block[0],
        (nphi + 2*g + block[1] - 1) // block[1],
        (nz + 1 + 2*g + block[2] - 1) // block[2],
    )
    hlld_z_kernel(
        grid, block,
        (state.get("rho"), state.get("rho_vr"), state.get("rho_vphi"),
         state.get("rho_vz"), state.get("E"),
         state.get("Br"), state.get("Bphi"), state.get("Bz"),
         fluxes["rho"], fluxes["vr"], fluxes["vphi"], fluxes["vz"],
         fluxes["E"], fluxes["Br"], fluxes["Bphi"],
         np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g))
    )


if __name__ == "__main__":
    print("hlld_z_kernel ready:", hlld_z_kernel is not None)
