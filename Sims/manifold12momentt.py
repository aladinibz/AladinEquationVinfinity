import cupy as cp

gyrotropic_12_kernel = cp.RawKernel(r'''
#define N_FEAT 12
#define MAX_NEWTON 4
#define RES_TOL 5e-4f

__device__ __forceinline__ void eval_phi(float cp, float c2, float p[N_FEAT]) {
    float cp2=cp*cp, cp3=cp2*cp, cp4=cp2*cp2;
    float c4=c2*c2, c6=c4*c2;
    float inv_dp=1.f/(1.f+cp2), inv_dpe=1.f/(1.f+c2);
    p[0]=1.f; p[1]=cp; p[2]=c2; p[3]=cp2;
    p[4]=cp*inv_dp; p[5]=cp*c2*inv_dpe;
    p[6]=c4; p[7]=cp2*c2; p[8]=cp4;
    p[9]=cp3*inv_dp*inv_dp; p[10]=c6*inv_dpe; p[11]=cp2*c4*inv_dpe;
}

__constant__ float np[16]={-6.5f,-5.5f,-4.5f,-3.5f,-2.5f,-1.5f,-0.5f,0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,-7.5f};
__constant__ float wp[16]={0.001f,0.004f,0.012f,0.03f,0.07f,0.13f,0.19f,0.19f,0.13f,0.07f,0.03f,0.012f,0.004f,0.001f,0.0003f,0.0003f};
__constant__ float nq[12]={0.2f,0.55f,1.0f,1.5f,2.1f,2.8f,3.6f,4.5f,5.6f,6.8f,8.2f,10.0f};
__constant__ float wq[12]={0.06f,0.12f,0.16f,0.17f,0.15f,0.12f,0.08f,0.05f,0.03f,0.015f,0.008f,0.003f};

__device__ float safe_exp(float x) { return __expf(fminf(fmaxf(x,-60.f),60.f)); }

__device__ void moments_only(const float* lambda, float ap, float ape, float* M) {
    float Z=0.f, M1[N_FEAT]={0};
    for(int ip=0;ip<16;++ip){
        float cp=np[ip], wpar=wp[ip];
        for(int iq=0;iq<12;++iq){
            float c2=nq[iq]*nq[iq];
            float w=wpar*wq[iq]*nq[iq];
            float p[N_FEAT]; eval_phi(cp,c2,p);
            float expo=-ap*cp*cp-ape*c2;
            #pragma unroll
            for(int k=0;k<N_FEAT;++k) expo+=lambda[k]*p[k];
            float weight=safe_exp(expo)*w;
            Z+=weight;
            #pragma unroll
            for(int i=0;i<N_FEAT;++i) M1[i]+=weight*p[i];
        }
    }
    float invZ=1.f/fmaxf(Z,1e-40f);
    #pragma unroll
    for(int i=0;i<N_FEAT;++i) M[i]=M1[i]*invZ;
}

extern "C" __global__
__launch_bounds__(128,4)
void gyrotropic_12_kernel(
    const float* __restrict__ M_target,
    const float* __restrict__ n_in,
    float* __restrict__ lambda_io,
    float* __restrict__ n_out,
    float* __restrict__ upar_out,
    float* __restrict__ Ppar_out,
    float* __restrict__ Pperp_out,
    float* __restrict__ qpar_out,
    float* __restrict__ qperp_out,
    int* __restrict__ status,
    int Ncells, float mass, float thermal_scale,
    float alpha_par, float alpha_perp)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=Ncells) return;

    const float* Mt = M_target + idx*N_FEAT;
    float* lam = lambda_io + idx*N_FEAT;

    float lambda[N_FEAT];
    // forced safe start
    lambda[0]=0.f; lambda[1]=0.f; lambda[2]=0.f; lambda[3]=0.f;
    lambda[4]=0.f; lambda[5]=0.f;
    lambda[6]=-0.2f; lambda[7]=-0.2f; lambda[8]=-0.2f;
    lambda[9]=0.f; lambda[10]=-0.1f; lambda[11]=-0.1f;

    float a_par=fmaxf(alpha_par,0.1f);
    float a_perp=fmaxf(alpha_perp,0.1f);

    float Mfinal[N_FEAT];
    moments_only(lambda, a_par, a_perp, Mfinal);   // always produce something

    int flag = 0;   // 0 = used safe start (for now)

    // write lambda
    for(int k=0;k<N_FEAT;++k) lam[k]=lambda[k];
    status[idx]=flag;

    float n_phys = n_in[idx];
    float upar = Mfinal[1];
    float cpar2 = fmaxf(Mfinal[3] - upar*upar, 1e-6f);
    float cperp2 = fmaxf(Mfinal[2], 1e-6f);

    float Ppar  = mass * n_phys * cpar2;
    float Pperp = 0.5f * mass * n_phys * cperp2;
    float qpar  = mass * n_phys * Mfinal[4] * thermal_scale;
    float qperp = mass * n_phys * Mfinal[5] * thermal_scale;

    n_out[idx]=n_phys;
    upar_out[idx]=upar;
    Ppar_out[idx]=Ppar;
    Pperp_out[idx]=Pperp;
    qpar_out[idx]=qpar;
    qperp_out[idx]=qperp;
}
''', 'gyrotropic_12_kernel')

# test
Ncells = 4096
M_target = cp.zeros((Ncells,12), dtype=cp.float32)
M_target[:,0] = 1.0
M_target[:,2] = 1.0
M_target[:,3] = 1.0
n_in = cp.ones(Ncells, dtype=cp.float32)
lambda_io = cp.zeros((Ncells,12), dtype=cp.float32)
n_out=cp.empty(Ncells,dtype=cp.float32)
upar=cp.empty(Ncells,dtype=cp.float32)
Ppar=cp.empty(Ncells,dtype=cp.float32)
Pperp=cp.empty(Ncells,dtype=cp.float32)
qpar=cp.empty(Ncells,dtype=cp.float32)
qperp=cp.empty(Ncells,dtype=cp.float32)
status=cp.zeros(Ncells,dtype=cp.int32)

threads=128
blocks=(Ncells+127)//128
gyrotropic_12_kernel((blocks,),(threads,),
    (M_target, n_in, lambda_io, n_out, upar, Ppar, Pperp, qpar, qperp,
     status, Ncells, 1.0, 1.0, 0.5, 0.5))

print("status unique:", cp.unique(status))
print("Ppar:", Ppar[:4])
print("Pperp:", Pperp[:4])
print("qpar/qperp max:", float(cp.max(cp.abs(qpar))), float(cp.max(cp.abs(qperp))))
