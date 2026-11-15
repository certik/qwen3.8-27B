// Shared between the C host code (main.c) and the CUDA kernels (kernels.cu).
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// One MLX affine 8-bit matrix (group size 64), on the device, in the checkpoint's own layout:
// W[N][K] bytes (the U32 tensor [N][K/4] read as bytes), S/B[N][K/64] bf16 scales and biases.
// Element (n,k) = W[n][k] * S[n][k/64] + B[n][k/64].
typedef struct {
    const uint8_t* W;
    const uint16_t* S;
    const uint16_t* B;
    int N, K;
} qmat;

// The residual stream and its RMSNorm, kept fused into the GEMVs:
//   x   residual [H]
//   xw  x * (weight of the RMSNorm that comes next), the input of the next GEMV
//   ss  sum of x^2, as 64-bit fixed point (2^-24 units) so that the atomic accumulation over blocks
//       is exact and deterministic. Two slots: a producer (G_RESID GEMV, embed) accumulates into
//       ss[k], the consuming GEMV reads ss[k], multiplies its result by rsqrt(ss/H + eps) (the same
//       as feeding it rmsnorm(x) * weight) and zeroes ss[1-k] for the next producer.
typedef struct {
    float *x, *xw;
    unsigned long long* ss;
    int k;
    int H;
    float eps;
} resid;

// All activations are fp32. "pos" and "tok" are device pointers, so that a whole token's forward
// pass can be captured once as a CUDA graph and replayed.

// x = embed[tok]; xw = x * w; ss = sum x^2
void k_embed(qmat E, const int* tok, const float* w, resid r, cudaStream_t s);
// y = rsqrt(ss/H + eps) * (W xw)
void k_gemv_norm(qmat m, resid r, float* y, cudaStream_t s);
// Same, for a [gate; up] matrix whose rows are interleaved in pairs (g0 g1 u0 u1 g2 g3 u2 u3 ...):
// out[i] = silu(gate_i) * up_i, i < N/2.
void k_gemv_swiglu(qmat m, resid r, float* out, cudaStream_t s);
// x += W in; then xw = x * wnext and ss = sum x^2 (for the next RMSNorm, weight wnext).
void k_gemv_resid(qmat m, const float* in, const float* wnext, resid r, cudaStream_t s);
// Plain y = W x (no norm)
void k_gemv(qmat m, const float* x, float* y, cudaStream_t s);

// Linear attention (Gated DeltaNet), one token: conv1d + SiLU, q/k L2 norm, gates, the delta-rule
// state update and the gated RMSNorm, in one kernel (one 512-thread block per value head).
// y = in_proj output [qkv (C) | z (nv*128) | b (nv) | a (nv)], C = 2*nk*128 + nv*128.
// conv_state: [2][3][C] (double-buffered by position parity). ssm: [nv][128 k][128 v]. out: [nv*128].
void k_gdn(const float* y, float* conv_state, const float* conv_w, const float* A_log, const float* dt_bias,
           const float* norm_w, float* ssm, float* out, const int* pos, int nv, int nk, float eps, cudaStream_t s);

// Full attention, one token (a cluster of 8 blocks per KV head): q/k RMSNorm, RoPE, KV-cache append,
// softmax(q K^T / 16) V, output gate.
// y = [q|gate per head (nq*512) | k (nkv*256) | v (nkv*256)]. out: [nq*256].
typedef struct {
    const float *qn_w, *kn_w;
    float *kcache, *vcache;    // [max_ctx][nkv*256]
    int nq, nkv, rot;
    float theta, eps;
} attn_params;
void k_attention(const float* y, attn_params a, float* out, const int* pos, cudaStream_t s);

// tok = argmax(x); hist[pos] = tok (hist: host-mapped memory, so the host can poll for it); pos += 1.
// pv/pi: scratch [256]; counter kept at 0; zero: the RMSNorm slot to clear for the next token.
void k_argmax(const float* x, int n, int* tok, int* pos, volatile int* hist, float* pv, int* pi, unsigned* counter,
              unsigned long long* zero, cudaStream_t s);


#ifdef __cplusplus
}
#endif
