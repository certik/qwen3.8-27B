// kernels.h — one C function per distinct math op in the MLX reference.
//
// Architecture: Qwen 3.5 dense/MoE, 8-bit affine MLX-quantized weights.
//
// Storage conventions:
//   bf16          : uint16_t (top 16 bits of an IEEE-754 f32)
//   q8 weight     : uint32 packed-byte tensor; per row: 4 uint8s per uint32
//                   in little-endian order (low byte = first element). For
//                   a Linear with shape (out=N, in=K), the storage shape is
//                   (N, K/4) uint32.
//   q8 scales/bias: bf16 per (out_row, in_group) where in_group = K/64.
//                   Dequant for element [n, k]:
//                       byte = (uint8) (W[n, k/4] >> (8 * (k % 4)))
//                       x    = (float)byte * to_f32(scales[n, k/64])
//                                          + to_f32(biases[n, k/64])
//   q8 switch     : same as q8 but with a leading num_experts dimension.

#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>
#include "utils/bf16.h"

// ---------- embed / RMSNorm / argmax ----------------------------------------

// Y[m, d] = dequant(E_q8[ids[m], d])
// E_q8.weight   : uint32  [V, D/4]   (low byte first)
// E_q8.scales   : bf16    [V, D/64]
// E_q8.biases   : bf16    [V, D/64]
void embed_dequant_gather_q8(const int32_t* ids,
                             const uint32_t* W,
                             const bf16* S, const bf16* B,
                             bf16* Y,
                             uint32_t M, uint32_t D);

// y[m, d] = w[d] * x[m, d] / sqrt(eps + mean_d(x[m, d]^2))
// If W is NULL, treat weight as ones (used by GatedDeltaNet internals).
void rmsnorm_bf16(const bf16* X, const bf16* W, bf16* Y,
                  uint32_t M, uint32_t D, float eps);

// argmax along last axis. `logits` is bf16 [N]; out_idx[0] = argmax.
void argmax_bf16(const bf16* logits, int32_t* out_idx, uint32_t N);

// ---------- linear (q8) -----------------------------------------------------

// Y[m, n] = bias[n] + Σ_k dequant(W_q8[n, k]) * X[m, k]
// `B` is the optional bias (can be NULL). For Qwen 3.5 there are no
// explicit biases on quantized linears (attention_bias=False, MoE uses
// SwitchLinear with bias=False).
//   X      : bf16    [M, K]
//   W      : uint32  [N, K/4]
//   S      : bf16    [N, K/64]
//   Bq     : bf16    [N, K/64]
//   Y      : bf16    [M, N]
//   bias   : bf16    [N]      (may be NULL)
// Precision is the frozen MLX 0.31.2 M4 profile: matrix shapes use BF16
// dequantized weights, split-K partial stores and strided reduction; vector shapes retain
// FP32 affine arithmetic. One scalar loop implementation handles every M.
void linear_q8_bf16(const bf16* X,
                    const uint32_t* W, const bf16* S, const bf16* Bq,
                    const bf16* bias,
                    bf16* Y,
                    uint32_t M, uint32_t K, uint32_t N);

// Like linear_q8_bf16 but with a leading num_experts dimension on W/S/Bq.
// For each token l and each of its K_top expert indices e = expert_ids[l, kt],
//   Y[l, kt, n] = Σ_k dequant(W[e, n, k]) * X[l, k]
//   X            : bf16    [L, K]
//   W            : uint32  [E, N, K/4]
//   S            : bf16    [E, N, K/64]
//   Bq           : bf16    [E, N, K/64]
//   expert_ids   : i32     [L, K_top]
//   Y            : bf16    [L, K_top, N]
void linear_q8_gather_bf16(const bf16* X,
                           const uint32_t* W, const bf16* S, const bf16* Bq,
                           const int32_t* expert_ids,
                           bf16* Y,
                           uint32_t L, uint32_t K_top,
                           uint32_t K, uint32_t N);

// ---------- attention -------------------------------------------------------

// Partial RoPE: rotate first `D_rot` dims of each head using freqs[d/2]
// and per-position phase. `traditional=False` style (split halves).
// X layout: [L, H, D]; only the first D_rot dims of each (l, h) row are
// modified. `inv_freqs[d/2]` = 1 / (base ** (d / D_rot)) for d in [0, D_rot/2).
// `q_offset` is the absolute position of X[0, *, *] in the KV cache.
void rope_partial_inplace_bf16(bf16* X,
                               const float* inv_freqs,
                               uint32_t L, uint32_t H, uint32_t D,
                               uint32_t D_rot, uint32_t q_offset);

// GQA SDPA with optional causal mask (causal=1).
//   Q    : bf16    [Lq, Nq, D]
//   K    : bf16    [Lk, Nkv, D]
//   V    : bf16    [Lk, Nkv, D]
//   OUT  : bf16    [Lq, Nq, D]
// Lq tokens at absolute positions [q_offset, q_offset + Lq).
// Nq is divisible by Nkv (GQA). Causal mask applies if causal != 0.
// Native unsupported fused shapes (notably D=256, Lq>8) use BF16 scaled Q,
// score and probability boundaries, matching MLX's actual fallback.
void sdpa_gqa_bf16(const bf16* Q, const bf16* K, const bf16* V,
                   bf16* OUT,
                   uint32_t Lq, uint32_t Lk,
                   uint32_t Nq, uint32_t Nkv, uint32_t D,
                   uint32_t q_offset, float scale, int causal);

// Element-wise: out = sigmoid(gate) * x
// `gate` and `x` are both bf16 [N]; `out` may alias `x`.
void elemwise_sigmoid_mul_bf16(const bf16* gate, const bf16* x,
                               bf16* out, uint32_t N);

// ---------- MoE -------------------------------------------------------------

// For each token, softmax(logits) then pick top-K indices and (renormalized)
// scores. logits is bf16 [L, E]; out_idx is i32 [L, K]; out_w is f32 [L, K].
// If `renorm` is nonzero, divide each row of out_w by its sum.
// Native BF16 probabilities/renormalization are widened into out_w.
// Equal-score top-K ties retain higher indices (stable ascending partition,
// last K entries); this is separate from first-index greedy argmax.
void softmax_topk_bf16(const bf16* logits,
                       int32_t* out_idx, float* out_w,
                       uint32_t L, uint32_t E, uint32_t K,
                       int renorm);

// BF16 products followed by the native strided BF16 reduction.
void expert_mix_bf16(const bf16* per_expert_out, const float* topk_w,
                     bf16* out, uint32_t L, uint32_t K_top, uint32_t H);

// FFN SwiGLU: BF16 SiLU store, then BF16 product.
void silu_mul_bf16(const bf16* gate, const bf16* up, bf16* out, uint32_t N);

// RMSNormGated: FP32 SiLU/product, only the output is stored as BF16.
void silu_mul_f32_gate_bf16(const bf16* gate, const bf16* up, bf16* out, uint32_t N);

// In-place: x += d
void residual_add_bf16(bf16* x, const bf16* d, uint32_t N);

// ---------- GatedDeltaNet (linear-attention) ---------------------------------

// Depthwise causal Conv1d, kernel=K (typically 4).
//   X      : bf16  [L, C]     (new tokens; concatenated downstream of state)
//   state  : bf16  [K-1, C]   (carry from previous step; updated on return)
//   W      : bf16  [C, K]     (per-channel weights; layout matches MLX after sanitize)
//   Y      : bf16  [L, C]
// Causal semantics: Y[l, c] = Σ_{j=0..K-1} W[c, j] * input_at(l - (K-1) + j, c)
// where input_at(t, c) reads `state` for t < 0 and `X` for t >= 0. On return
// the last K-1 input rows are written back to `state` (so subsequent calls
// continue the convolution).
void conv1d_depthwise_causal_bf16(const bf16* X, bf16* state,
                                  const bf16* W, bf16* Y,
                                  uint32_t L, uint32_t C, uint32_t K);

// SiLU in place: x[i] = x[i] * sigmoid(x[i])
void silu_inplace_bf16(bf16* X, uint32_t N);

// RMSNorm with weight=None and an external multiplier:
//   y[..., d] = scalar * x[..., d] / sqrt(eps + mean_d(x^2))
// X is treated as [N, D] (the inner-D axis is the normalized one).
void rmsnorm_scale_bf16(const bf16* X, bf16* Y,
                        uint32_t N, uint32_t D,
                        float eps, float scalar);

// compute_g: g[i] = exp(-exp(A_log[i]) * softplus(a[i] + dt_bias[i]))
//   a       : bf16  [L, Hv]
//   A_log   : bf16  [Hv]
//   dt_bias : bf16  [Hv]
// BF16 a+dt and softplus stores precede FP32 exponent arithmetic.
//   g       : f32   [L, Hv] (never narrow before recurrence)
void compute_g_f32(const bf16* a, const bf16* A_log, const bf16* dt_bias,
                   float* g, uint32_t L, uint32_t Hv);

// Sigmoid in place: x[i] = 1/(1 + exp(-x[i]))
void sigmoid_inplace_bf16(bf16* X, uint32_t N);

// One step of the gated-delta-net recurrence for a *single* token (Lq=1).
// Hk = num_k_heads, Hv = num_v_heads, Hv % Hk == 0. Each k_head serves
// Hv/Hk consecutive v_heads. State has dtype f32 throughout.
//   q       : bf16  [Hk, Dk]
//   k       : bf16  [Hk, Dk]
//   v       : bf16  [Hv, Dv]
//   g       : f32   [Hv]
//   beta    : bf16  [Hv]
//   state   : f32   [Hv, Dv, Dk]   (in/out)
//   y       : bf16  [Hv, Dv]
void gated_delta_step_bf16(const bf16* q, const bf16* k, const bf16* v,
                           const float* g, const bf16* beta,
                           float* state,
                           bf16* y,
                           uint32_t Hk, uint32_t Hv,
                           uint32_t Dk, uint32_t Dv);

#endif
