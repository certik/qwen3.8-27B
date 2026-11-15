// CUDA kernels for Qwen3.8-27B (MLX 8-bit) single-token decode. fp32 activations,
// weights stay in the checkpoint's q8 + bf16 scale/bias format.
//
// One token = 5 kernels per layer (4 GEMVs + one Gated DeltaNet or attention kernel) + 3:
// RMSNorm, residual adds and SwiGLU are fused into the GEMVs (see model.h). All kernels use
// programmatic dependent launch, so that each kernel's independent loads overlap the previous kernel.
#include "model.h"

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

static __device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
static __device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

// After each block has written its partial result: returns true in exactly one block, the last
// one to finish, which then sees all partials (read them with __ldcg). Resets the counter.
static __device__ bool last_block(unsigned* counter, unsigned nblocks) {
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(counter, 1u) == nblocks - 1;
        if (last) *counter = 0;
    }
    __syncthreads();
    return last;
}

// Programmatic dependent launch (PDL): every kernel is launched with programmatic stream
// serialization and calls pdl_trigger() at its start, so that the next kernel's blocks can become
// resident right away, and pdl_wait() before it reads anything the previous kernel wrote. Loads that
// do not depend on the previous kernel (the first step of a GEMV's weights, the recurrent and conv
// state, the first key of attention) are issued before pdl_wait(). pdl_wait() returns once the
// previous kernel has completed and its memory is visible. Env QWEN_PDL=0 disables PDL.
static __device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
static __device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

static int use_pdl() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("QWEN_PDL");
        v = e ? atoi(e) : 1;
    }
    return v;
}

template <typename... KArgs, typename... Args>
static void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = use_pdl();
    cfg.attrs = at;
    cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, ((KArgs)args)...);
    if (e != cudaSuccess) { fprintf(stderr, "cudaLaunchKernelEx: %s\n", cudaGetErrorString(e)); exit(1); }
}

static void check_launch(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "CUDA launch error in %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}

// ---------------------------------------------------------------- embedding

// Fixed-point accumulation of the RMSNorm sum of squares (see resid in model.h).
constexpr float SS_SCALE = 16777216.f;   // 2^24
static __device__ void ss_add(const resid& r, float v) {
    atomicAdd(r.ss + r.k, (unsigned long long)(v * SS_SCALE + 0.5f));
}
static __device__ float ss_get(const resid& r) { return (float)((double)r.ss[r.k] * (1.0 / SS_SCALE)); }


__global__ void __launch_bounds__(256) embed_kernel(qmat E, const int* tok, const float* w, resid r) {
    pdl_trigger();
    pdl_wait();
    const size_t row = (size_t)min(max(*tok, 0), E.N - 1);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E.K / 4) return;
    float ss = 0.f;
    for (int k = 4 * i; k < 4 * i + 4; k++) {
        const size_t g = row * (E.K / 64) + k / 64;
        const float v = (float)E.W[row * E.K + k] * bf(E.S[g]) + bf(E.B[g]);
        r.x[k] = v;
        r.xw[k] = v * w[k];
        ss += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) ss_add(r, ss);
}

extern "C" void k_embed(qmat E, const int* tok, const float* w, resid r, cudaStream_t s) {
    launch(embed_kernel, (E.K / 4 + 255) / 256, 256, 0, s, E, tok, w, r);
    check_launch("embed");
}

// ---------------------------------------------------------------- q8 GEMV
// The kernel of ../kernels/q8_gemv.cu (see the design notes there): a block of T threads computes
// 4 consecutive rows, 16-byte streaming loads, exact byte -> float conversion, bias term via the
// sum of x. Plus fused epilogues.

constexpr int ROWS = 4;
enum { G_PLAIN, G_NORM, G_SWIGLU, G_RESID };

struct gemv_args {
    const uint8_t* W;
    const uint16_t* S;
    const uint16_t* B;
    const float* x;
    float* y;
    int K;
    const float* wnext;
    resid r;
};

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
    uint4 r;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ float u8_to_f32(uint32_t w, int i) {
    return __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440 | i)) - 8388608.0f;
}
__device__ __forceinline__ float dot4(uint32_t w, float4 x) {
    return u8_to_f32(w, 0) * x.x + u8_to_f32(w, 1) * x.y + u8_to_f32(w, 2) * x.z + u8_to_f32(w, 3) * x.w;
}

// MINB: blocks per SM the register budget must allow (16 -> 64 registers). UNR: main-loop unroll,
// i.e. steps of loads in flight per thread.
// The read-only inputs are separate __restrict__ parameters. In a standalone harness, reading them from
// a struct instead made nvcc spread a step's ROWS weight loads apart, 20% slower; here it made no
// measurable difference, but this keeps the loads together (see experiments/README.md).
template <int T, int MODE, int MINB, int UNR, int PRE>
__global__ void __launch_bounds__(T, MINB)
gemv_kernel(const uint8_t* __restrict__ W, const uint16_t* __restrict__ Sc, const uint16_t* __restrict__ Bi,
            const float* __restrict__ X, gemv_args a) {
    const int row0 = blockIdx.x * ROWS;
    const int t = threadIdx.x;
    const int K = a.K, nchunk = K / 16, G = K / 64;
    pdl_trigger();
    // One step of the main loop, on weights already loaded into q/sc/bi.
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.f;
    auto step = [&](int c, const uint4* q, const uint16_t* sc, const uint16_t* bi) {
        const float4* xp = reinterpret_cast<const float4*>(X + c * 16);
        const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1), x2 = __ldg(xp + 2), x3 = __ldg(xp + 3);
        const float xs = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w)
                       + (x2.x + x2.y + x2.z + x2.w) + (x3.x + x3.y + x3.z + x3.w);
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const float p = dot4(q[r].x, x0) + dot4(q[r].y, x1) + dot4(q[r].z, x2) + dot4(q[r].w, x3);
            acc[r] += bf(sc[r]) * p + bf(bi[r]) * xs;
        }
    };
    auto load = [&](int c, uint4* q, uint16_t* sc, uint16_t* bi) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            q[r] = ld_stream(reinterpret_cast<const uint4*>(W + (size_t)(row0 + r) * K) + c);
            const size_t g = (size_t)(row0 + r) * G + (c >> 2);
            sc[r] = __ldg(Sc + g);
            bi[r] = __ldg(Bi + g);
        }
    };
    {   // The first PRE steps' weights do not depend on the previous kernel: request them before
        // waiting for it (PDL). K >= 16 * T * PRE, so every thread has PRE first chunks.
        uint4 q[PRE][ROWS];
        uint16_t sc[PRE][ROWS], bi[PRE][ROWS];
#pragma unroll
        for (int u = 0; u < PRE; u++) load(t + u * T, q[u], sc[u], bi[u]);
        pdl_wait();
#pragma unroll
        for (int u = 0; u < PRE; u++) step(t + u * T, q[u], sc[u], bi[u]);
    }
#pragma unroll UNR
    for (int c = t + PRE * T; c < nchunk; c += T) {
        uint4 q[ROWS];
        uint16_t sc[ROWS], bi[ROWS];
        load(c, q, sc, bi);
        step(c, q, sc, bi);
    }
    __shared__ float red[ROWS][T / 32];
    __shared__ float fin[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float v = acc[r];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if ((t & 31) == 0) red[r][t >> 5] = v;
    }
    __syncthreads();
    float v = 0.f;
    if (t < ROWS) {
        for (int w = 0; w < T / 32; w++) v += red[t][w];
        if (MODE == G_NORM || MODE == G_SWIGLU) {
            v *= rsqrtf(ss_get(a.r) / a.r.H + a.r.eps);
            // Zero the other slot for the next producer. (Not at the start of the kernel: a global
            // store before the main loop makes these GEMVs ~10% slower.)
            if (blockIdx.x == 0 && t == 0) a.r.ss[1 - a.r.k] = 0;
        }
    }
    if (MODE == G_PLAIN || MODE == G_NORM) {
        if (t < ROWS) a.y[row0 + t] = v;
    } else if (MODE == G_SWIGLU) {   // rows: gate 2b, gate 2b+1, up 2b, up 2b+1
        if (t < ROWS) fin[t] = v;
        __syncthreads();
        if (t < 2) {
            const float g = fin[t];
            a.y[blockIdx.x * 2 + t] = g * sigmoidf_(g) * fin[t + 2];
        }
    } else {   // G_RESID
        if (t < ROWS) {
            const float xn = a.r.x[row0 + t] + v;
            a.r.x[row0 + t] = xn;
            a.r.xw[row0 + t] = xn * a.wnext[row0 + t];
            fin[t] = xn * xn;
        }
        __syncthreads();
        if (t == 0) ss_add(a.r, (fin[0] + fin[1]) + (fin[2] + fin[3]));
    }
}

template <int MODE>
static void launch_gemv(qmat m, const gemv_args& a, cudaStream_t s) {
    if (m.K % 64 || m.N % ROWS) { fprintf(stderr, "gemv: unsupported shape %d x %d\n", m.N, m.K); exit(1); }
    const int nb = m.N / ROWS;
    // Matrices with few rows (out_proj, down: 1280 blocks) fill only ~10 of the 16 block slots per SM
    // anyway, so they can use up to ~100 registers for a deeper unroll (4 steps of loads in flight).
    if (nb <= 10 * 132) launch(gemv_kernel<64, MODE, 10, 4, 1>, nb, 64, 0, s, a.W, a.S, a.B, a.x, a);
    else launch(gemv_kernel<64, MODE, 16, 2, 1>, nb, 64, 0, s, a.W, a.S, a.B, a.x, a);
    check_launch("gemv");
}

static gemv_args make_args(qmat m, const float* x, float* y) {
    gemv_args a = {};
    a.W = m.W; a.S = m.S; a.B = m.B; a.K = m.K; a.x = x; a.y = y;
    return a;
}

extern "C" void k_gemv(qmat m, const float* x, float* y, cudaStream_t s) {
    launch_gemv<G_PLAIN>(m, make_args(m, x, y), s);
}
extern "C" void k_gemv_norm(qmat m, resid r, float* y, cudaStream_t s) {
    gemv_args a = make_args(m, r.xw, y);
    a.r = r;
    launch_gemv<G_NORM>(m, a, s);
}
extern "C" void k_gemv_swiglu(qmat m, resid r, float* out, cudaStream_t s) {
    gemv_args a = make_args(m, r.xw, out);
    a.r = r;
    launch_gemv<G_SWIGLU>(m, a, s);
}
extern "C" void k_gemv_resid(qmat m, const float* in, const float* wnext, resid r, cudaStream_t s) {
    gemv_args a = make_args(m, in, nullptr);
    a.r = r;
    a.wnext = wnext;
    launch_gemv<G_RESID>(m, a, s);
}

// ---------------------------------------------------------------- linear attention (Gated DeltaNet)
// One block of 512 threads per value head. Thread (part, w, lane) = (tid / 128, tid / 32 % 4, tid % 32)
// holds state[k = 32w .. 32w+31][v = 32 part + lane] in registers: the 128 x 128 state is read once
// and written once, and all reductions stay inside the block.

constexpr int GD = 128;   // key/value head dim

__global__ void __launch_bounds__(512)
gdn_kernel(const float* y, float* conv_state, const float* conv_w, const float* A_log, const float* dt_bias,
           const float* norm_w, float* ssm, float* out, const int* pos, int nv, int nk, float eps) {
    const int hv = blockIdx.x, rep = nv / nk, hk = hv / rep;
    const int tid = threadIdx.x, part = tid >> 7, w = (tid >> 5) & 3, lane = tid & 31, dv = part * 32 + lane;
    const int C = 2 * nk * GD + nv * GD;
    pdl_trigger();
    // Everything except y (the in_proj output) is ready before the previous kernel finishes: load the
    // recurrent state, conv state, conv weights and gate parameters first, then wait.
    float* Sp = ssm + (size_t)hv * GD * GD + (size_t)(w * 32) * GD + dv;
    float s[32];
#pragma unroll
    for (int j = 0; j < 32; j++) s[j] = Sp[j * GD];
    const int p = *pos;   // only changed by argmax, at the end of the token
    const float* st_in = conv_state + (size_t)(p & 1) * 3 * C;
    float* st_out = conv_state + (size_t)((p + 1) & 1) * 3 * C;
    __shared__ float qs[GD], ks[GD], vs[GD], red[4][4][32], sums[8];

    // Causal conv (4 taps) + SiLU: thread tid < 384 handles q (key head hk), k, or v (value head hv)
    // channel tid % 128. q/k are shared by `rep` value heads; only the first of them writes their state.
    const int c = tid < GD ? hk * GD + tid : tid < 2 * GD ? nk * GD + hk * GD + (tid - GD) : 2 * nk * GD + hv * GD + (tid - 2 * GD);
    const bool has_ch = tid < 3 * GD, writes = tid >= 2 * GD || hv % rep == 0;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f;
    float4 wc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (has_ch) {
        s0 = st_in[c];
        s1 = st_in[C + c];
        s2 = st_in[2 * C + c];
        wc = reinterpret_cast<const float4*>(conv_w)[c];   // oldest input first
    }
    const float alog = A_log[hv], dtb = dt_bias[hv], nw = norm_w[dv];
    pdl_wait();
    const float* ab = y + C + nv * GD;   // b [nv], then a [nv]
    const float zz = y[C + hv * GD + dv], bb = ab[hv], aa = ab[nv + hv];
    if (has_ch) {
        const float xin = y[c];
        float v = wc.x * s0 + wc.y * s1 + wc.z * s2 + wc.w * xin;
        v = v * sigmoidf_(v);
        if (writes) {
            st_out[c] = s1;
            st_out[C + c] = s2;
            st_out[2 * C + c] = xin;
        }
        // q and k: sums of squares for the L2 norm (warps 0-3: q, 4-7: k)
        float sq = v * v;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0 && tid < 2 * GD) sums[tid >> 5] = sq;
        if (tid < GD) qs[tid] = v;
        else if (tid < 2 * GD) ks[tid - GD] = v;
        else vs[tid - 2 * GD] = v;
    }
    // Gates: decay g and write strength beta.
    const float x = aa + dtb;
    const float softplus = x > 20.f ? x : log1pf(expf(x));
    const float g = expf(-expf(alog) * softplus);
    const float beta = 1.f / (1.f + expf(-bb));
    __syncthreads();
    // q, k: L2-normalized; q also scaled by 1/sqrt(128). Applied on the fly below.
    const float qn = rsqrtf((sums[0] + sums[1]) + (sums[2] + sums[3]) + 1e-6f) * rsqrtf((float)GD);
    const float kn = rsqrtf((sums[4] + sums[5]) + (sums[6] + sums[7]) + 1e-6f);

    float a = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j++) a += s[j] * ks[w * 32 + j];
    red[part][w][lane] = a;
    __syncthreads();
    const float kvm = ((red[part][0][lane] + red[part][1][lane]) + (red[part][2][lane] + red[part][3][lane])) * kn;
    const float delta = (vs[dv] - g * kvm) * beta;
    const float dk = delta * kn;
    float o = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        const float sn = s[j] * g + ks[w * 32 + j] * dk;
        Sp[j * GD] = sn;
        o += sn * qs[w * 32 + j];
    }
    __syncthreads();
    red[part][w][lane] = o;
    __syncthreads();
    if (w == 0) {   // one warp per part: finish o for 32 value dims, then the gated RMSNorm over all 128
        o = ((red[part][0][lane] + red[part][1][lane]) + (red[part][2][lane] + red[part][3][lane])) * qn;
        float s2 = o * o;
        for (int off = 16; off > 0; off >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, off);
        if (lane == 0) sums[part] = s2;
    }
    __syncthreads();
    if (w == 0) {
        const float tot = (sums[0] + sums[1]) + (sums[2] + sums[3]);
        out[hv * GD + dv] = o * rsqrtf(tot / GD + eps) * nw * (zz * sigmoidf_(zz));
    }
}

extern "C" void k_gdn(const float* y, float* conv_state, const float* conv_w, const float* A_log,
                      const float* dt_bias, const float* norm_w, float* ssm, float* out, const int* pos, int nv,
                      int nk, float eps, cudaStream_t s) {
    launch(gdn_kernel, nv, 512, 0, s, y, conv_state, conv_w, A_log, dt_bias, norm_w, ssm, out, pos, nv, nk, eps);
}

// ---------------------------------------------------------------- full attention
// One cluster of 8 blocks (8 warps each) per KV head. The 64 warps of a cluster take every 64th key
// and keep an online softmax (max, sum, 256-dim accumulator for each of the `group` query heads that
// share the KV head) in registers; lane l owns dims 8l..8l+7. Each block merges its 8 warps in
// shared memory; then the 8 blocks merge through distributed shared memory, each block finishing
// 32 of the 256 dims. Output gate: out = attn * sigmoid(gate).

constexpr int HD = 256;
constexpr int MAXG = 6;   // query heads per KV head (Qwen3.8-27B: 24 / 4)
constexpr int ACL = 8;    // blocks per cluster

static size_t attn_smem(int G) { return (size_t)(G * HD + HD + 8 * G * HD + G * HD + 2 * G) * sizeof(float); }

// One warp: RMSNorm (weight w) + rotate-half RoPE of one 256-vector src -> dst (shared memory).
// Lane l holds dims l + 32j (j = 0..7), so the RoPE partner of dim d < 32 (d + 32) is in the same
// lane; this assumes rot == 64. cs/sn: this lane's cos/sin.
static __device__ void warp_norm_rope(const float* src, float* dst, const float* w, float cs, float sn, float eps) {
    const int lane = threadIdx.x & 31;
    float v[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        v[j] = src[lane + 32 * j];
        ss += v[j] * v[j];
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float r = rsqrtf(ss / HD + eps);
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] *= r * w[lane + 32 * j];
    const float a = v[0], b = v[1];
    v[0] = a * cs - b * sn;
    v[1] = b * cs + a * sn;
#pragma unroll
    for (int j = 0; j < 8; j++) dst[lane + 32 * j] = v[j];
}

__global__ void __cluster_dims__(ACL, 1, 1) __launch_bounds__(HD)
attention_kernel(const float* y, attn_params a, float* out, const int* pos) {
    constexpr int G = MAXG;
    cg::cluster_group cluster = cg::this_cluster();
    extern __shared__ float sm[];
    float* qs = sm;                   // [G][256]
    float* kb = qs + G * HD;          // [256] k of the current position
    float* wacc = kb + HD;            // [8 warps][G][256]
    float* bacc = wacc + 8 * G * HD;  // [G][256] this block's merged accumulator
    float* bml = bacc + G * HD;       // [G][2] this block's max and sum
    const int rank = blockIdx.x % ACL, kh = blockIdx.x / ACL;
    const int d = threadIdx.x, warp = d >> 5, lane = d & 31;
    pdl_trigger();
    const int p = *pos, L = p + 1;   // pos is only changed by argmax, at the end of the token
    const size_t stride = (size_t)a.nkv * HD;   // one position in the KV cache
    const double inv_freq = pow((double)a.theta, -2.0 * lane / a.rot);
    float sn, cs;
    sincosf((float)fmod((double)p * inv_freq, 6.283185307179586), &sn, &cs);
    // This warp's first key (if it is an earlier position, it does not depend on the previous kernel).
    constexpr int NW = ACL * 8;
    const int tfirst = rank * 8 + warp;
    float4 k0, k1, v0, v1;
    if (tfirst < p) {
        const float* kp = a.kcache + tfirst * stride + kh * HD + lane * 8;
        const float* vp = a.vcache + tfirst * stride + kh * HD + lane * 8;
        k0 = *(const float4*)kp; k1 = *(const float4*)(kp + 4);
        v0 = *(const float4*)vp; v1 = *(const float4*)(vp + 4);
    }
    pdl_wait();

    // q of the G heads, and k of the current position (every block needs it; rank 0 appends k, v to the cache).
    if (warp < G) warp_norm_rope(y + (size_t)(kh * G + warp) * 2 * HD, qs + warp * HD, a.qn_w, cs, sn, a.eps);
    const float* vnew = y + (size_t)a.nq * 2 * HD + a.nkv * HD + kh * HD;
    if (warp == G) {
        warp_norm_rope(y + (size_t)a.nq * 2 * HD + kh * HD, kb, a.kn_w, cs, sn, a.eps);
        if (rank == 0)
            for (int j = lane; j < HD; j += 32) {
                a.kcache[p * stride + kh * HD + j] = kb[j];
                a.vcache[p * stride + kh * HD + j] = vnew[j];
            }
    }
    __syncthreads();

    float qr[G][8], acc[G][8], m[G], l[G];
#pragma unroll
    for (int i = 0; i < G; i++) {
        m[i] = -INFINITY;
        l[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            qr[i][j] = qs[i * HD + lane * 8 + j] * 0.0625f;   // 1/sqrt(256)
            acc[i][j] = 0.f;
        }
    }
    // Keys t = gw, gw + 64, ... for global warp gw; the current position comes from shared memory / y.
    auto load = [&](int t, float4& k0, float4& k1, float4& v0, float4& v1) {
        const float* kp = t == p ? kb + lane * 8 : a.kcache + t * stride + kh * HD + lane * 8;
        const float* vp = t == p ? vnew + lane * 8 : a.vcache + t * stride + kh * HD + lane * 8;
        k0 = *(const float4*)kp; k1 = *(const float4*)(kp + 4);
        v0 = *(const float4*)vp; v1 = *(const float4*)(vp + 4);
    };
    int t = tfirst;
    if (t == p) load(t, k0, k1, v0, v1);
    for (; t < L; t += NW) {
        const float kk[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
        const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        if (t + NW < L) load(t + NW, k0, k1, v0, v1);   // prefetch this warp's next key
        float sc[G];
#pragma unroll
        for (int i = 0; i < G; i++) {
            float dot = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) dot += qr[i][j] * kk[j];
            sc[i] = dot;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (int i = 0; i < G; i++) sc[i] += __shfl_xor_sync(0xffffffffu, sc[i], o);
#pragma unroll
        for (int i = 0; i < G; i++) {
            const float mn = fmaxf(m[i], sc[i]);
            const float corr = __expf(m[i] - mn), pr = __expf(sc[i] - mn);
            m[i] = mn;
            l[i] = l[i] * corr + pr;
#pragma unroll
            for (int j = 0; j < 8; j++) acc[i][j] = acc[i][j] * corr + pr * vv[j];
        }
    }
    // Merge the block's 8 warps (warps without keys have m = -inf, l = 0).
    __shared__ float wm[8][G], wl[8][G];
#pragma unroll
    for (int i = 0; i < G; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) wacc[(warp * G + i) * HD + lane * 8 + j] = acc[i][j];
        if (lane == 0) { wm[warp][i] = m[i]; wl[warp][i] = l[i]; }
    }
    __syncthreads();
    for (int i = 0; i < G; i++) {
        float M = -INFINITY;
        for (int w = 0; w < 8; w++) M = fmaxf(M, wm[w][i]);
        float num = 0.f, den = 0.f;
        if (M != -INFINITY)
            for (int w = 0; w < 8; w++) {
                const float e = __expf(wm[w][i] - M);
                num += e * wacc[(w * G + i) * HD + d];
                den += e * wl[w][i];
            }
        bacc[i * HD + d] = num;
        if (d == 0) { bml[2 * i] = M; bml[2 * i + 1] = den; }
    }
    // Merge the cluster's 8 blocks: this block finishes dims rank*32 .. rank*32+31 of every head.
    cluster.sync();
    if (d < G * 32) {
        const int i = d / 32, dim = rank * 32 + (d % 32);
        float mr[ACL], M = -INFINITY;
#pragma unroll
        for (int r = 0; r < ACL; r++) {
            mr[r] = cluster.map_shared_rank(bml, r)[2 * i];
            M = fmaxf(M, mr[r]);
        }
        float num = 0.f, den = 0.f;
#pragma unroll
        for (int r = 0; r < ACL; r++) {
            if (mr[r] == -INFINITY) continue;
            const float e = __expf(mr[r] - M);
            num += e * cluster.map_shared_rank(bacc, r)[i * HD + dim];
            den += e * cluster.map_shared_rank(bml, r)[2 * i + 1];
        }
        const int h = kh * G + i;
        out[h * HD + dim] = num / den * sigmoidf_(y[(size_t)h * 2 * HD + HD + dim]);
    }
    cluster.sync();   // keep shared memory alive until the other blocks have read it
}

extern "C" void k_attention(const float* y, attn_params a, float* out, const int* pos, cudaStream_t s) {
    if (a.nq / a.nkv != MAXG || a.rot != 64) {
        fprintf(stderr, "attention: unsupported group %d / rot %d\n", a.nq / a.nkv, a.rot);
        exit(1);
    }
    const size_t smem = attn_smem(MAXG);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    launch(attention_kernel, a.nkv * ACL, HD, smem, s, y, a, out, pos);
}

// ---------------------------------------------------------------- argmax (+ advance position)

constexpr int AM_BLOCKS = 256;

// One float4 per thread (n % 4 == 0, n / 4 <= 256 * 256), block max, then the last block merges.
__global__ void __launch_bounds__(256) argmax_kernel(const float* x, int n, int* tok, int* pos, volatile int* hist,
                                                     float* pv, int* pi, unsigned* counter, unsigned long long* zero) {
    __shared__ float sv[256];
    __shared__ int si[256];
    pdl_trigger();
    pdl_wait();
    const int t = threadIdx.x, i4 = blockIdx.x * blockDim.x + t;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    if (i4 < n / 4) {
        const float4 v = reinterpret_cast<const float4*>(x)[i4];
        const float e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; j++)
            if (e[j] > best) { best = e[j]; bi = 4 * i4 + j; }
    }
    auto reduce = [&]() {
        sv[t] = best;
        si[t] = bi;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (t < st) {
                const float v = sv[t + st];
                const int j = si[t + st];
                if (v > sv[t] || (v == sv[t] && j < si[t])) { sv[t] = v; si[t] = j; }
            }
            __syncthreads();
        }
    };
    reduce();
    if (t == 0) { pv[blockIdx.x] = sv[0]; pi[blockIdx.x] = si[0]; }
    if (last_block(counter, gridDim.x)) {
        best = __ldcg(pv + t);
        bi = __ldcg(pi + t);
        __syncthreads();
        reduce();
        if (t == 0) {
            *tok = si[0];
            hist[*pos] = si[0];
            *pos += 1;
            *zero = 0;
        }
    }
}

extern "C" void k_argmax(const float* x, int n, int* tok, int* pos, volatile int* hist, float* pv, int* pi,
                         unsigned* counter, unsigned long long* zero, cudaStream_t s) {
    if (n % 4 || n / 4 > AM_BLOCKS * 256) { fprintf(stderr, "argmax: unsupported n %d\n", n); exit(1); }
    launch(argmax_kernel, AM_BLOCKS, 256, 0, s, x, n, tok, pos, hist, pv, pi, counter, zero);
}
