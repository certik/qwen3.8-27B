// Experiment: one token's 257 q8 GEMVs (the real fused Qwen3.8-27B shapes, 27.2 GB) as ONE persistent
// kernel, compared with the same work as 257 kernels in a CUDA graph. Not used by ../qwen; see
// README.md in this directory for the results and what they mean.
//
// Structure: one block per SM (cooperative launch), WARPS warps each. Weights are repacked into tiles of
// 4 rows x KP bytes (+ their scales/biases), contiguous, so one TMA bulk copy fetches a tile. Each SM owns
// a contiguous slice of the row groups of every matrix; its tiles are dealt round-robin to its warps.
// Each warp owns an NBUF-slot shared-memory ring and is its own producer: lane 0 issues cp.async.bulk
// (+ mbarrier) NBUF-1 jobs ahead. Weights never depend on the previous matrix, so the ring keeps
// streaming across matrix boundaries; only the math waits for the grid-wide "stage done" counter.
// x is staged into shared memory per stage (permuted so the warp's float4 reads have no bank conflicts).
// Every spin loop has a watchdog (trap) so that a bug cannot hang a shared GPU.
//
// Build: nvcc -O3 -arch=sm_90a -std=c++17 persistent_gemv.cu -o persistent_gemv   (needs ~55 GB of GPU memory)
// Switches (timing experiments only; results are then wrong): -DEXP=4 no math, 8 no weight streaming,
// 2 no grid barrier; -DWARPS_=, -DNBUF_=, -DKP_= (KP must divide 5120, 6144 and 17408: 512 or 1024).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#ifndef EXP
#define EXP 0
#endif
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
static __device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ float u8f(uint32_t w, int i) { return __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440 | i)) - 8388608.0f; }
__device__ __forceinline__ float dot4(uint32_t w, float4 x) { return u8f(w,0)*x.x + u8f(w,1)*x.y + u8f(w,2)*x.z + u8f(w,3)*x.w; }

struct Stage { const uint8_t* W; const uint16_t* S; const uint16_t* B; int N, K; };
#ifndef WARPS_
#define WARPS_ 8
#endif
#ifndef NBUF_
#define NBUF_ 4
#endif
#ifndef KP_
#define KP_ 1024
#endif
constexpr int WARPS = WARPS_, NBUF = NBUF_, KP = KP_;   // job = 4 rows x KP bytes (one tile)
constexpr int TILE = 4 * KP, TS = 4 * (KP / 64);   // weight bytes, scales per tile
struct Buf { uint8_t w[TILE]; uint16_t s[TS]; uint16_t b[TS]; };
constexpr int XMAX = 17408, PMAX = 2048;

__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, unsigned n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(b)), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void bulk(void* dst, const void* src, unsigned bytes, uint64_t* b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, unsigned phase) {
    asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W%=; }" :: "r"(sa(b)), "r"(phase) : "memory"); }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

// Tiled layout (made by repack): tile (g, p) = rows 4g..4g+3, bytes p*KP..: [4 rows][KP] contiguous;
// scales/biases tile = [4 rows][KP/64].
// Each warp owns an NBUF-slot ring and is its own producer (lane 0 issues TMA bulk copies NBUF-1 jobs ahead).
// Watchdogs: every spin loop traps after ~2^28 polls instead of hanging the GPU.
__device__ __forceinline__ bool mbar_try(uint64_t* b, unsigned phase) {
    unsigned ok;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(sa(b)), "r"(phase) : "memory");
    return ok;
}
__device__ __forceinline__ void mbar_wait_wd(uint64_t* b, unsigned phase) {
    unsigned n = 0;
    while (!mbar_try(b, phase)) if (++n > (1u << 28)) asm volatile("trap;");
}
__global__ void __launch_bounds__(WARPS * 32, 1) mega(const Stage* __restrict__ st, int nst, unsigned* arrive,
                                                      const float* __restrict__ x, float* __restrict__ y) {
    extern __shared__ __align__(128) uint8_t smem[];
    Buf* bufs = reinterpret_cast<Buf*>(smem);
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + sizeof(Buf) * WARPS * NBUF);
    float* xs_ = reinterpret_cast<float*>(bars + WARPS * NBUF);
    float* part = xs_ + XMAX;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nb = gridDim.x, b = blockIdx.x;
    Buf* my = bufs + warp * NBUF;
    uint64_t* mb = bars + warp * NBUF;
    if (lane == 0) for (int i = 0; i < NBUF; i++) mbar_init(&mb[i], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();
    // per-stage tables for this block: first row group, number of row groups, parts per row (K / KP)
    __shared__ int tg0[512], tng[512], tP[512];
    for (int s = threadIdx.x; s < nst; s += blockDim.x) {
        const int ng = st[s].N / 4;
        const int g0 = (int)((long long)ng * b / nb), g1 = (int)((long long)ng * (b + 1) / nb);
        tg0[s] = g0; tng[s] = g1 - g0; tP[s] = st[s].K / KP;
    }
    __syncthreads();
    // job cursor: stage s, job j (= lg * P + p) within the block's jobs of stage s
    struct Cur { int s, lg, p; };
    auto first = [&](Cur& c, int s) {
        for (; s < nst; s++) if (warp < tng[s] * tP[s]) { c = {s, warp / tP[s], warp % tP[s]}; return; }
        c = {nst, 0, 0};
    };
    auto next = [&](Cur& c) {
        c.p += WARPS;
        const int P = tP[c.s];
        while (c.p >= P) { c.p -= P; c.lg++; }
        if (c.lg >= tng[c.s]) first(c, c.s + 1);
    };
    auto issue = [&](const Cur& c, int slot) {
        if (c.s >= nst || lane != 0 || (EXP & 8)) return;
        const Stage S = st[c.s];
        const int P = tP[c.s];
        const size_t tile = (size_t)(tg0[c.s] + c.lg) * P + c.p;
        mbar_expect(&mb[slot], TILE + 4 * TS);
        bulk(my[slot].w, S.W + tile * TILE, TILE, &mb[slot]);
        bulk(my[slot].s, S.S + tile * TS, 2 * TS, &mb[slot]);
        bulk(my[slot].b, S.B + tile * TS, 2 * TS, &mb[slot]);
    };
    Cur cur, pre;
    first(cur, 0);
    pre = cur;
    unsigned phase = 0;
    int slot = 0, pslot = 0;
    for (int i = 0; i < NBUF - 1; i++) { issue(pre, pslot); if (pre.s < nst) next(pre); pslot = (pslot + 1) % NBUF; }
    for (int s = 0; s < nst; s++) {
        const int K = st[s].K, g0 = tg0[s], g1 = g0 + tng[s];
        if (!(EXP & 2) && s > 0 && threadIdx.x == 0) {
            unsigned n = 0;
            while (ld_acquire(&arrive[s - 1]) < (unsigned)nb) { __nanosleep(20); if (++n > (1u << 26)) asm volatile("trap;"); }
        }
        __syncthreads();
        // x staged permuted: float4 k (0..3) of 16-element chunk c lives at float4 index (c/32)*128 + k*32 + c%32,
        // so the 32 lanes of a warp (consecutive c) read consecutive float4s: no bank conflicts.
        {   // all loads first (one L2 round trip), then the stores
            constexpr int MAXI = (XMAX / 4 + WARPS * 32 - 1) / (WARPS * 32);
            float4 v[MAXI];
#pragma unroll
            for (int u = 0; u < MAXI; u++) {
                const int i = threadIdx.x + u * WARPS * 32;
                if (i < K / 4) v[u] = __ldcg(reinterpret_cast<const float4*>(x) + i);
            }
#pragma unroll
            for (int u = 0; u < MAXI; u++) {
                const int i = threadIdx.x + u * WARPS * 32;
                if (i < K / 4) { const int c = i >> 2, k = i & 3; reinterpret_cast<float4*>(xs_)[(c >> 5) * 128 + k * 32 + (c & 31)] = v[u]; }
            }
        }
        for (int i = threadIdx.x; i < 4 * (g1 - g0); i += blockDim.x) part[i] = 0.f;
        __syncthreads();
        while (cur.s == s) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue(pre, pslot);
            if (pre.s < nst) next(pre);
            pslot = (pslot + 1) % NBUF;
            if (!(EXP & 8)) mbar_wait_wd(&mb[slot], (phase >> slot) & 1);
            phase ^= 1u << slot;
            const Buf& B = my[slot];
            const int lg = cur.lg, p = cur.p;
            float acc[4] = {0, 0, 0, 0};
            if (!(EXP & 4)) {
#pragma unroll
                for (int u = 0; u < KP / 512; u++) {
                    const int c = lane + 32 * u;
                    const int cg_ = p * (KP / 16) + c;   // global chunk index
                    const float4* xp = reinterpret_cast<const float4*>(xs_) + (cg_ >> 5) * 128 + (cg_ & 31);
                    const float4 x0 = xp[0], x1 = xp[32], x2 = xp[64], x3 = xp[96];
                    const float xsum = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w) + (x2.x + x2.y + x2.z + x2.w) + (x3.x + x3.y + x3.z + x3.w);
#pragma unroll
                    for (int r = 0; r < 4; r++) {
                        const uint4 q = reinterpret_cast<const uint4*>(B.w + r * KP)[c];
                        acc[r] += bf(B.s[r * (KP / 64) + (c >> 2)]) * (dot4(q.x, x0) + dot4(q.y, x1) + dot4(q.z, x2) + dot4(q.w, x3))
                                + bf(B.b[r * (KP / 64) + (c >> 2)]) * xsum;
                    }
                }
#pragma unroll
                for (int r = 0; r < 4; r++)
                    for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
                if (lane < 4) atomicAdd(&part[lg * 4 + lane], acc[lane]);
            }
            __syncwarp();
            slot = (slot + 1) % NBUF;
            next(cur);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < 4 * (g1 - g0); i += blockDim.x) y[g0 * 4 + i] = part[i];
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(&arrive[s], 1u);
    }
}

// row-major -> tiled
__global__ void repack(const uint8_t* W, const uint16_t* S, const uint16_t* B, uint8_t* Wt, uint16_t* St, uint16_t* Bt, int N, int K) {
    const size_t n = (size_t)N * K;
    const int P = K / KP;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n / 16; i += (size_t)gridDim.x * blockDim.x) {
        const size_t e = i * 16; const int row = (int)(e / K), k = (int)(e % K);
        const size_t tile = (size_t)(row / 4) * P + k / KP;
        reinterpret_cast<uint4*>(Wt + tile * TILE + (row % 4) * KP + (k % KP))[0] = reinterpret_cast<const uint4*>(W + e)[0];
        if (k % 64 == 0) {
            const size_t gi = (size_t)row * (K / 64) + k / 64, go = tile * TS + (row % 4) * (KP / 64) + (k % KP) / 64;
            St[go] = S[gi]; Bt[go] = B[gi];
        }
    }
}

__global__ void __launch_bounds__(64, 16) one(const uint8_t* __restrict__ W, const uint16_t* __restrict__ Sc, const uint16_t* __restrict__ Bi, int K,
                                             const float* __restrict__ x, float* __restrict__ y) {
    const int t = threadIdx.x, row0 = blockIdx.x * 4, nchunk = K / 16, Gk = K / 64;
    float acc[4] = {0, 0, 0, 0};
#pragma unroll 2
    for (int c = t; c < nchunk; c += 64) {
        uint4 q[4]; uint16_t sc[4], bi[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            q[r] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)(row0 + r) * K) + c);
            const size_t g = (size_t)(row0 + r) * Gk + (c >> 2);
            sc[r] = __ldg(Sc + g); bi[r] = __ldg(Bi + g);
        }
        const float4* xp = reinterpret_cast<const float4*>(x + c * 16);
        const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1), x2 = __ldg(xp + 2), x3 = __ldg(xp + 3);
        const float xs = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w) + (x2.x + x2.y + x2.z + x2.w) + (x3.x + x3.y + x3.z + x3.w);
#pragma unroll
        for (int r = 0; r < 4; r++)
            acc[r] += bf(sc[r]) * (dot4(q[r].x, x0) + dot4(q[r].y, x1) + dot4(q[r].z, x2) + dot4(q[r].w, x3)) + bf(bi[r]) * xs;
    }
    __shared__ float red[4][2];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        float v = acc[r];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if ((t & 31) == 0) red[r][t >> 5] = v;
    }
    __syncthreads();
    if (t < 4) y[row0 + t] = red[t][0] + red[t][1];
}

__global__ void fill(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = (uint8_t)((i * 2654435761u + seed) >> 13);
}
__global__ void fillbf(uint16_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (uint16_t)(0x3c00 + (((i * 2246822519u + seed) >> 20) & 0xff));
}

int main() {
    std::vector<Stage> hs; size_t bytes = 0;
    auto add = [&](int N, int K) {
        Stage s{}; s.N = N; s.K = K; size_t g = (size_t)N * (K / 64);
        uint8_t* w; uint16_t *sc, *b;
        CK(cudaMalloc(&w, (size_t)N * K)); CK(cudaMalloc(&sc, g * 2)); CK(cudaMalloc(&b, g * 2));
        fill<<<1024, 256>>>(w, (size_t)N * K, hs.size()); fillbf<<<256, 256>>>(sc, g, 1); fillbf<<<256, 256>>>(b, g, 2);
        s.W = w; s.S = sc; s.B = b; hs.push_back(s); bytes += (size_t)N * K + 4 * g;
    };
    for (int l = 0; l < 64; l++) { bool full = (l % 4 == 3); add(full ? 14336 : 16480, 5120); add(5120, 6144); add(34816, 5120); add(5120, 17408); }
    add(248320, 5120);
    const int nst = (int)hs.size();
    std::vector<Stage> ht = hs;
    for (auto& t : ht) {
        size_t g = (size_t)t.N * (t.K / 64); uint8_t* w; uint16_t *sc, *b;
        CK(cudaMalloc(&w, (size_t)t.N * t.K)); CK(cudaMalloc(&sc, g * 2)); CK(cudaMalloc(&b, g * 2));
        repack<<<4096, 256>>>(t.W, t.S, t.B, w, sc, b, t.N, t.K);
        t.W = w; t.S = sc; t.B = b;
    }
    Stage* dst; CK(cudaMalloc(&dst, nst * sizeof(Stage))); CK(cudaMemcpy(dst, ht.data(), nst * sizeof(Stage), cudaMemcpyHostToDevice));
    unsigned* arrive; CK(cudaMalloc(&arrive, nst * 4));
    float *x, *y; CK(cudaMalloc(&x, 17408 * 4)); CK(cudaMalloc(&y, 248320 * 4));
    { std::vector<float> hx(17408); for (int i = 0; i < 17408; i++) hx[i] = sinf(0.37f * i); CK(cudaMemcpy(x, hx.data(), 17408 * 4, cudaMemcpyHostToDevice)); }
    CK(cudaDeviceSynchronize());
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaStream_t s; CK(cudaStreamCreate(&s));

    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (auto& t : hs) one<<<t.N / 4, 64, 0, s>>>(t.W, t.S, t.B, t.K, x, y);
    CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    float best = 1e9;
    for (int i = 0; i < 6; i++) { cudaEventRecord(e0, s); cudaGraphLaunch(ge, s); cudaEventRecord(e1, s); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); best = fminf(best, ms); }
    printf("%d stages, %.2f GB\n257 kernels (graph):  %.3f ms/token  %.2f TB/s\n", nst, bytes / 1e9, best, bytes / (best * 1e-3) / 1e12);
    std::vector<float> ref(248320); CK(cudaMemcpy(ref.data(), y, 248320 * 4, cudaMemcpyDeviceToHost));

    const size_t smem = sizeof(Buf) * WARPS * NBUF + sizeof(uint64_t) * WARPS * NBUF + (XMAX + PMAX) * sizeof(float);
    CK(cudaFuncSetAttribute(mega, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    best = 1e9;
    for (int i = 0; i < 6; i++) {
        CK(cudaMemsetAsync(arrive, 0, nst * 4, s));
        int nst_ = nst; void* args[] = {&dst, &nst_, &arrive, &x, &y};
        cudaEventRecord(e0, s);
        CK(cudaLaunchCooperativeKernel((void*)mega, nsm, WARPS * 32, args, smem, s));
        cudaEventRecord(e1, s); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1); best = fminf(best, ms);
    }
    printf("persistent + TMA ring: %.3f ms/token  %.2f TB/s  (smem %zu B/SM)\n", best, bytes / (best * 1e-3) / 1e12, smem);
    std::vector<float> got(248320); CK(cudaMemcpy(got.data(), y, 248320 * 4, cudaMemcpyDeviceToHost));
    double md = 0, mr = 0; for (int i = 0; i < 248320; i++) { md = fmax(md, fabs(got[i] - ref[i])); mr = fmax(mr, fabs(ref[i])); }
    printf("lm_head output: max |diff| %g (max |ref| %g)\n", md, mr);
}
