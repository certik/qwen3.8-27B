# Hack: h1 + the bodies of the two gemv kernels replaced by src-cuda's gemv_kernel (adapted to our arguments).
# $T1PDL=1: CUDA's PDL prologue (trigger, first chunk loaded before the wait)
import sys, re, os, subprocess
D = os.path.dirname(os.path.abspath(__file__))
p = sys.argv[1]
subprocess.check_call(['python3', os.path.join(D, 'h1.py'), p])
s = open(p).read()
pdl = os.environ.get('T1PDL') == '1'
common = r'''
__device__ __forceinline__ uint4 cu_ld_stream(const uint4* p) {
    uint4 r;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ float cu_bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ float cu_u8(uint32_t w, int i) { return __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440 | i)) - 8388608.0f; }
__device__ __forceinline__ float cu_dot4(uint32_t w, float4 x) { return cu_u8(w, 0) * x.x + cu_u8(w, 1) * x.y + cu_u8(w, 2) * x.z + cu_u8(w, 3) * x.w; }
// src-cuda's gemv_kernel main loop: acc[r] = row (row0 + r) . X, block of T threads
template <int T, int UNR, int PRE, bool PDL>
__device__ __forceinline__ void cu_gemv_rows(const uint8_t* __restrict__ W, const uint16_t* __restrict__ Sc,
        const uint16_t* __restrict__ Bi, const float* __restrict__ X, int K, int row0, float* acc) {
    const int t = threadIdx.x, nchunk = K / 16, G = K / 64;
    for (int r = 0; r < 4; r++) acc[r] = 0.f;
    auto step = [&](int c, const uint4* q, const uint16_t* sc, const uint16_t* bi) {
        const float4* xp = reinterpret_cast<const float4*>(X + c * 16);
        const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1), x2 = __ldg(xp + 2), x3 = __ldg(xp + 3);
        const float xs = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w) + (x2.x + x2.y + x2.z + x2.w) + (x3.x + x3.y + x3.z + x3.w);
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const float pp = cu_dot4(q[r].x, x0) + cu_dot4(q[r].y, x1) + cu_dot4(q[r].z, x2) + cu_dot4(q[r].w, x3);
            acc[r] += cu_bf(sc[r]) * pp + cu_bf(bi[r]) * xs;
        }
    };
    auto load = [&](int c, uint4* q, uint16_t* sc, uint16_t* bi) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
            q[r] = cu_ld_stream(reinterpret_cast<const uint4*>(W + (size_t)(row0 + r) * K) + c);
            const size_t g = (size_t)(row0 + r) * G + (c >> 2);
            sc[r] = __ldg(Sc + g);
            bi[r] = __ldg(Bi + g);
        }
    };
    {
        uint4 q[PRE][4]; uint16_t sc[PRE][4], bi[PRE][4];
#pragma unroll
        for (int u = 0; u < PRE; u++) load(t + u * T, q[u], sc[u], bi[u]);
        if (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
        for (int u = 0; u < PRE; u++) step(t + u * T, q[u], sc[u], bi[u]);
    }
#pragma unroll UNR
    for (int c = t + PRE * T; c < nchunk; c += T) {
        uint4 q[4]; uint16_t sc[4], bi[4];
        load(c, q, sc, bi);
        step(c, q, sc, bi);
    }
}
'''
norm_body = r'''{
    const int row0 = blockIdx.x * 4, t = threadIdx.x, K = __scalar_args.__dim_w_0 * 4, kk = __scalar_args.k;
    PDLSTART
    float acc[4];
    cu_gemv_rows<64, NUNR, 1, PDLB>((const uint8_t*)w, (const uint16_t*)s, (const uint16_t*)b, xw, K, row0, acc);
    __shared__ float red[4][2];
    __shared__ float fin[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        float v = acc[r];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if ((t & 31) == 0) red[r][t >> 5] = v;
    }
    __syncthreads();
    float v = 0.f;
    if (t < 4) {
        v = red[t][0] + red[t][1];
        v *= rsqrtf((float)((double)((const unsigned long long*)ssp)[kk - 1] * (1.0 / 16777216.0)) / __scalar_args.__dim_xw_0 + __scalar_args.eps);
    }
    if (!__scalar_args.swiglu) {
        if (t < 4) y[row0 + t] = v;
    } else {
        if (t < 4) fin[t] = v;
        __syncthreads();
        if (t < 2) { const float g = fin[t]; y[blockIdx.x * 2 + t] = g * (1.f / (1.f + __expf(-g))) * fin[t + 2]; }
    }
}
'''
resid_body = r'''{
    const int row0 = blockIdx.x * 4, t = threadIdx.x, K = __scalar_args.__dim_w_0 * 4, kk = __scalar_args.k;
    PDLSTART
    float acc[4];
    cu_gemv_rows<64, RUNR, 1, PDLB>((const uint8_t*)w, (const uint16_t*)s, (const uint16_t*)b, v, K, row0, acc);
    __shared__ float red[4][2];
    __shared__ float fin[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        float vv = acc[r];
        for (int o = 16; o > 0; o >>= 1) vv += __shfl_xor_sync(0xffffffffu, vv, o);
        if ((t & 31) == 0) red[r][t >> 5] = vv;
    }
    __syncthreads();
    if (t < 4) {
        const float xn = x[row0 + t] + (red[t][0] + red[t][1]);
        x[row0 + t] = xn;
        xw[row0 + t] = xn * wnext[row0 + t];
        fin[t] = xn * xn;
    }
    __syncthreads();
    if (t == 0) {
        unsigned long long* sl = (unsigned long long*)ssp;
        if (blockIdx.x == 0) sl[2 - kk] = 0;
        atomicAdd(&sl[kk - 1], (unsigned long long)(((fin[0] + fin[1]) + (fin[2] + fin[3])) * 16777216.f + 0.5f));
    }
}
'''
pdlstart = 'asm volatile("griddepcontrol.launch_dependents;" ::: "memory");' if pdl else 'asm volatile("griddepcontrol.wait;" ::: "memory");'
def body(t):
    return (t.replace('PDLSTART', pdlstart).replace('PDLB', 'true' if pdl else 'false')
             .replace('NUNR', os.environ.get('T1NU', '2')).replace('RUNR', os.environ.get('T1RU', '4')))
out = []; first = True
for k in re.split(r'(?=extern "C" __global__)', s):
    head = k.split('\n')[0]
    for name, bdy, mb in (('gemv_norm(', norm_body, os.environ.get('T1NB', '16')), ('gemv_resid(', resid_body, os.environ.get('T1RB', '10'))):
        if name in head:
            k = re.sub(r'__launch_bounds__\(64, \d+\)', '__launch_bounds__(64, %s)' % mb, k, 1)
            a = k.index('\n{\n'); e = k.index('\n}\n', a)
            k = k[:a] + '\n' + body(bdy) + k[e + 3:]
            if first: k = common + k; first = False
    out.append(k)
open(p, 'w').write(''.join(out))
