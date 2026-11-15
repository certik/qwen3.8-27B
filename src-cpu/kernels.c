// kernels.c — naive scalar implementations.
//
// Implementation order matches kernels.h. Each kernel is written as plain
// loops with fp32 accumulators where it matters. No fusion beyond what the
// MLX reference does. No prefill/decode special paths.

#include "kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// ---------- helpers ---------------------------------------------------------

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float siluf(float x)    { return x * sigmoidf(x); }

static inline float round_bf16(float x) { return bf16_to_f32(f32_to_bf16(x)); }

// MLX's native BF16 Sigmoid evaluates the stable positive-exponential form
// with BF16 exp, sum, quotient and subtraction results.
static float sigmoid_bf16_value(float x) {
    float e = round_bf16(expf(fabsf(x)));
    float y = round_bf16(1.0f / round_bf16(1.0f + e));
    return x < 0 ? y : round_bf16(1.0f - y);
}

static float sum_squares_f32(const bf16* x, uint32_t n) {
    // Keep FP32 storage/arithmetic without the width-dependent error of a
    // long left-to-right sum. A denominator error can cross a BF16 rounding
    // boundary and then be amplified by a large learned RMS weight.
    float sum = 0.0f, correction = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float value = bf16_to_f32(x[i]);
        float term = value*value - correction;
        float next = sum + term;
        correction = (next-sum) - term;
        sum = next;
    }
    return sum;
}

static float sum_strided_bf16(const bf16* partials, uint32_t n) {
    // Native strided BF16 sum: small reductions have up to eight BF16
    // lanes, then a BF16 fold. Larger reductions accumulate in 32 BF16
    // lanes before the SIMD sum's FP32 reduction and BF16 result.
    uint32_t lanes = n < 32 ? (n < 8 ? n : 8) : 32;
    float values[32] = {0};
    for (uint32_t lane = 0; lane < lanes; lane++)
        for (uint32_t p = lane; p < n; p += lanes)
            values[lane] = round_bf16(values[lane] + bf16_to_f32(partials[p]));
    if (n < 32) {
        float total = values[0];
        for (uint32_t lane = 1; lane < lanes; lane++) total = round_bf16(total + values[lane]);
        return total;
    }
    for (uint32_t stride = 16; stride; stride /= 2)
        for (uint32_t lane = 0; lane < stride; lane++) values[lane] += values[lane+stride];
    return round_bf16(values[0]);
}

// Read a single 8-bit element from a packed uint32 q8 row at column k.
static inline uint8_t q8_byte(const uint32_t* row, uint32_t k) {
    uint32_t w = row[k >> 2];
    return (uint8_t)((w >> (8 * (k & 3))) & 0xffu);
}

// Dequantize element [n, k] of a (N, K) q8 tensor. Group size is 64.
static inline float q8_deq(const uint32_t* W, const bf16* S, const bf16* Bq,
                           uint32_t n, uint32_t k, uint32_t Kpack)
{
    uint8_t q = q8_byte(W + (size_t)n * Kpack, k);
    uint32_t g = k >> 6;
    uint32_t Kgroups = Kpack * 4 / 64;
    float s = bf16_to_f32(S[(size_t)n * Kgroups + g]);
    float b = bf16_to_f32(Bq[(size_t)n * Kgroups + g]);
    return (float)q * s + b;
}

// ---------- embed_dequant_gather --------------------------------------------

void embed_dequant_gather_q8(const int32_t* ids,
                             const uint32_t* W,
                             const bf16* S, const bf16* B,
                             bf16* Y,
                             uint32_t M, uint32_t D)
{
    uint32_t Kpack   = D / 4;
    uint32_t Kgroups = D / 64;
    for (uint32_t m = 0; m < M; m++) {
        int32_t id = ids[m];
        const uint32_t* w_row = W + (size_t)id * Kpack;
        const bf16*     s_row = S + (size_t)id * Kgroups;
        const bf16*     b_row = B + (size_t)id * Kgroups;
        bf16* yrow = Y + (size_t)m * D;
        for (uint32_t k = 0; k < D; k++) {
            uint8_t q = q8_byte(w_row, k);
            uint32_t g = k >> 6;
            float v = (float)q * bf16_to_f32(s_row[g]) + bf16_to_f32(b_row[g]);
            yrow[k] = f32_to_bf16(v);
        }
    }
}

// ---------- rmsnorm ---------------------------------------------------------

void rmsnorm_bf16(const bf16* X, const bf16* W, bf16* Y,
                  uint32_t M, uint32_t D, float eps)
{
    for (uint32_t m = 0; m < M; m++) {
        const bf16* xrow = X + (size_t)m * D;
        bf16*       yrow = Y + (size_t)m * D;
        float sq = sum_squares_f32(xrow, D);
        float rrms = 1.0f / sqrtf(sq / (float)D + eps);
        if (W) {
            for (uint32_t d = 0; d < D; d++) {
                float normalized = bf16_to_f32(f32_to_bf16(bf16_to_f32(xrow[d]) * rrms));
                float v = normalized * bf16_to_f32(W[d]);
                yrow[d] = f32_to_bf16(v);
            }
        } else {
            for (uint32_t d = 0; d < D; d++) {
                float v = bf16_to_f32(xrow[d]) * rrms;
                yrow[d] = f32_to_bf16(v);
            }
        }
    }
}

void rmsnorm_scale_bf16(const bf16* X, bf16* Y,
                        uint32_t N, uint32_t D,
                        float eps, float scalar)
{
    scalar = round_bf16(scalar);
    for (uint32_t n = 0; n < N; n++) {
        const bf16* xrow = X + (size_t)n * D;
        bf16*       yrow = Y + (size_t)n * D;
        float sq = sum_squares_f32(xrow, D);
        float rrms = 1.0f / sqrtf(sq / (float)D + eps);
        for (uint32_t d = 0; d < D; d++) {
            float normalized = bf16_to_f32(f32_to_bf16(bf16_to_f32(xrow[d]) * rrms));
            yrow[d] = f32_to_bf16(normalized * scalar);
        }
    }
}

// ---------- linear (q8) -----------------------------------------------------

void linear_q8_bf16(const bf16* X,
                    const uint32_t* W, const bf16* S, const bf16* Bq,
                    const bf16* bias,
                    bf16* Y,
                    uint32_t M, uint32_t K, uint32_t N)
{
    uint32_t Kpack   = K / 4;
    uint32_t Kgroups = K / 64;
    // Frozen MLX 0.31.2 M4 arithmetic profile. These shape decisions change
    // precision (BF16 dequantized operands and split-K partial stores), not
    // the scalar implementation or its single-threaded execution.
    uint32_t vector_limit = K <= 2048 && N <= 2048 ? 18 :
                            K <= 4096 && N <= 4096 ? 12 : 10;
    int rounded_weights = M >= vector_limit;
    uint32_t parts = 1;
    if (rounded_weights) {
        uint64_t tiles = ((uint64_t)M+31)/32 * (((uint64_t)N+31)/32);
        parts = tiles < 512 ? (uint32_t)(512/tiles) : 1;
        if (parts > Kgroups) parts = Kgroups;
        while (parts > 1 && Kgroups % parts) parts--;
    }
    uint32_t groups_per_part = Kgroups/parts;
    bf16 partial_values[512];
    for (uint32_t m = 0; m < M; m++) {
        const bf16* xrow = X + (size_t)m * K;
        bf16*       yrow = Y + (size_t)m * N;
        for (uint32_t n = 0; n < N; n++) {
            const uint32_t* w_row = W + (size_t)n * Kpack;
            const bf16*     s_row = S + (size_t)n * Kgroups;
            const bf16*     b_row = Bq + (size_t)n * Kgroups;
            float total = bias ? bf16_to_f32(bias[n]) : 0.0f;
            for (uint32_t part = 0; part < parts; part++) {
                float acc = 0.0f;
                for (uint32_t g = part*groups_per_part; g < (part+1)*groups_per_part; g++) {
                    float s = bf16_to_f32(s_row[g]);
                    float b = bf16_to_f32(b_row[g]);
                    for (uint32_t kk = 0; kk < 64; kk++) {
                        uint32_t k = g * 64 + kk;
                        uint8_t q = q8_byte(w_row, k);
                        float w = (float)q*s + b;
                        if (rounded_weights) w = round_bf16(w);
                        acc += bf16_to_f32(xrow[k])*w;
                    }
                }
                if (parts > 1) partial_values[part] = f32_to_bf16(acc);
                else total += acc;
            }
            if (parts > 1) total += sum_strided_bf16(partial_values, parts);
            yrow[n] = f32_to_bf16(total);
        }
    }
}

void linear_q8_gather_bf16(const bf16* X,
                           const uint32_t* W, const bf16* S, const bf16* Bq,
                           const int32_t* expert_ids,
                           bf16* Y,
                           uint32_t L, uint32_t K_top,
                           uint32_t K, uint32_t N)
{
    uint32_t Kpack   = K / 4;
    uint32_t Kgroups = K / 64;
    size_t   per_exp_W = (size_t)N * Kpack;
    size_t   per_exp_S = (size_t)N * Kgroups;
    for (uint32_t l = 0; l < L; l++) {
        const bf16* xrow = X + (size_t)l * K;
        for (uint32_t kt = 0; kt < K_top; kt++) {
            int32_t e = expert_ids[(size_t)l * K_top + kt];
            const uint32_t* W_e = W + (size_t)e * per_exp_W;
            const bf16*     S_e = S + (size_t)e * per_exp_S;
            const bf16*     B_e = Bq + (size_t)e * per_exp_S;
            bf16* yrow = Y + ((size_t)l * K_top + kt) * N;
            for (uint32_t n = 0; n < N; n++) {
                const uint32_t* w_row = W_e + (size_t)n * Kpack;
                const bf16*     s_row = S_e + (size_t)n * Kgroups;
                const bf16*     b_row = B_e + (size_t)n * Kgroups;
                float acc = 0.0f;
                for (uint32_t g = 0; g < Kgroups; g++) {
                    float s = bf16_to_f32(s_row[g]);
                    float b = bf16_to_f32(b_row[g]);
                    for (uint32_t kk = 0; kk < 64; kk++) {
                        uint32_t k = g * 64 + kk;
                        uint8_t  q = q8_byte(w_row, k);
                        float    w = (float)q * s + b;
                        acc += bf16_to_f32(xrow[k]) * w;
                    }
                }
                yrow[n] = f32_to_bf16(acc);
            }
        }
    }
}

// ---------- RoPE ------------------------------------------------------------

void rope_partial_inplace_bf16(bf16* X,
                               const float* inv_freqs,
                               uint32_t L, uint32_t H, uint32_t D,
                               uint32_t D_rot, uint32_t q_offset)
{
    uint32_t Dh = D_rot / 2;
    for (uint32_t l = 0; l < L; l++) {
        float pos = (float)(q_offset + l);
        for (uint32_t h = 0; h < H; h++) {
            bf16* row = X + ((size_t)l * H + h) * D;
            for (uint32_t d = 0; d < Dh; d++) {
                float c = cosf(pos * inv_freqs[d]);
                float s = sinf(pos * inv_freqs[d]);
                float a = bf16_to_f32(row[d]);
                float b = bf16_to_f32(row[d + Dh]);
                row[d]      = f32_to_bf16(a * c - b * s);
                row[d + Dh] = f32_to_bf16(a * s + b * c);
            }
        }
    }
}

// ---------- SDPA (GQA, optional causal) -------------------------------------

void sdpa_gqa_bf16(const bf16* Q, const bf16* K, const bf16* V,
                   bf16* OUT,
                   uint32_t Lq, uint32_t Lk,
                   uint32_t Nq, uint32_t Nkv, uint32_t D,
                   uint32_t q_offset, float scale, int causal)
{
    uint32_t group = Nq / Nkv;
    int vector = Lq <= 8 && (uint64_t)Lq*group <= 32 &&
                 (D == 64 || D == 96 || D == 128 || D == 256);
    int full = Lq > 8 && (D == 64 || D == 80 || D == 128);
    int rounded_scores = !(vector || full);
    float input_scale = round_bf16(scale);
    for (uint32_t lq = 0; lq < Lq; lq++) {
        uint32_t q_abs = q_offset + lq;
        for (uint32_t hq = 0; hq < Nq; hq++) {
            uint32_t hkv = hq / group;
            const bf16* q_row = Q + ((size_t)lq * Nq + hq) * D;
            bf16*       o_row = OUT + ((size_t)lq * Nq + hq) * D;

            float max_s = -INFINITY;
            float* scores = (float*)malloc((size_t)Lk * sizeof(float));
            if (!scores) { fprintf(stderr, "attention allocation failed\n"); exit(1); }
            for (uint32_t lk = 0; lk < Lk; lk++) {
                if (causal && lk > q_abs) { scores[lk] = -INFINITY; continue; }
                const bf16* k_row = K + ((size_t)lk * Nkv + hkv) * D;
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; d++) {
                    float qv = bf16_to_f32(q_row[d]);
                    if (rounded_scores) qv = round_bf16(qv*input_scale);
                    dot += qv * bf16_to_f32(k_row[d]);
                }
                dot = rounded_scores ? round_bf16(dot) : dot*scale;
                scores[lk] = dot;
                if (dot > max_s) max_s = dot;
            }
            float sum = 0.0f;
            for (uint32_t lk = 0; lk < Lk; lk++) {
                float e = (scores[lk] == -INFINITY) ? 0.0f : expf(scores[lk] - max_s);
                scores[lk] = e;
                sum += e;
            }
            float inv_sum = 1.0f / sum;
            if (rounded_scores)
                for (uint32_t lk = 0; lk < Lk; lk++) scores[lk] = round_bf16(scores[lk]*inv_sum);
            for (uint32_t d = 0; d < D; d++) {
                float acc = 0.0f;
                for (uint32_t lk = 0; lk < Lk; lk++) {
                    if (scores[lk] == 0.0f) continue;
                    const bf16* v_row = V + ((size_t)lk * Nkv + hkv) * D;
                    acc += scores[lk] * bf16_to_f32(v_row[d]);
                }
                o_row[d] = f32_to_bf16(rounded_scores ? acc : acc*inv_sum);
            }
            free(scores);
        }
    }
}

// ---------- elementwise & residual -----------------------------------------

void elemwise_sigmoid_mul_bf16(const bf16* gate, const bf16* x,
                               bf16* out, uint32_t N)
{
    for (uint32_t i = 0; i < N; i++) {
        float g = bf16_to_f32(gate[i]);
        float v = bf16_to_f32(x[i]);
        out[i] = f32_to_bf16(sigmoid_bf16_value(g) * v);
    }
}

void silu_mul_bf16(const bf16* gate, const bf16* up, bf16* out, uint32_t N)
{
    for (uint32_t i = 0; i < N; i++) {
        float g = bf16_to_f32(gate[i]);
        float s = round_bf16(g * sigmoid_bf16_value(g));
        out[i] = f32_to_bf16(s * bf16_to_f32(up[i]));
    }
}

void silu_mul_f32_gate_bf16(const bf16* gate, const bf16* up, bf16* out, uint32_t N)
{
    for (uint32_t i = 0; i < N; i++) {
        float g = bf16_to_f32(gate[i]);
        float u = bf16_to_f32(up[i]);
        out[i] = f32_to_bf16(siluf(g) * u);
    }
}

void residual_add_bf16(bf16* x, const bf16* d, uint32_t N)
{
    for (uint32_t i = 0; i < N; i++) {
        x[i] = f32_to_bf16(bf16_to_f32(x[i]) + bf16_to_f32(d[i]));
    }
}

void silu_inplace_bf16(bf16* X, uint32_t N)
{
    for (uint32_t i = 0; i < N; i++) {
        float x = bf16_to_f32(X[i]);
        X[i] = f32_to_bf16(x * sigmoid_bf16_value(x));
    }
}

void sigmoid_inplace_bf16(bf16* X, uint32_t N)
{
    for (uint32_t i = 0; i < N; i++) {
        X[i] = f32_to_bf16(sigmoid_bf16_value(bf16_to_f32(X[i])));
    }
}

// ---------- argmax ----------------------------------------------------------

void argmax_bf16(const bf16* logits, int32_t* out_idx, uint32_t N)
{
    int32_t best_i = 0;
    float   best_v = bf16_to_f32(logits[0]);
    for (uint32_t i = 1; i < N; i++) {
        float v = bf16_to_f32(logits[i]);
        if (v > best_v) { best_v = v; best_i = (int32_t)i; }
    }
    out_idx[0] = best_i;
}

// ---------- softmax top-K (router) ------------------------------------------

void softmax_topk_bf16(const bf16* logits,
                       int32_t* out_idx, float* out_w,
                       uint32_t L, uint32_t E, uint32_t K,
                       int renorm)
{
    for (uint32_t l = 0; l < L; l++) {
        const bf16* row = logits + (size_t)l * E;
        // fp32 softmax (matches MLX precise=True)
        float max_x = -INFINITY;
        for (uint32_t e = 0; e < E; e++) {
            float v = bf16_to_f32(row[e]);
            if (v > max_x) max_x = v;
        }
        float* p = (float*)malloc((size_t)E * sizeof(float));
        float sum = 0.0f;
        for (uint32_t e = 0; e < E; e++) {
            float v = expf(bf16_to_f32(row[e]) - max_x);
            p[e] = v;
            sum += v;
        }
        float inv = 1.0f / sum;
        for (uint32_t e = 0; e < E; e++) p[e] = round_bf16(p[e]*inv);

        // Pick top-K (insertion sort by score; K is small).
        int32_t* idx = out_idx + (size_t)l * K;
        float*   w   = out_w   + (size_t)l * K;
        for (uint32_t k = 0; k < K; k++) { idx[k] = -1; w[k] = -INFINITY; }
        for (uint32_t e = 0; e < E; e++) {
            float v = p[e];
            // find smallest in (idx, w) and replace if v > it
            int   imin = 0;
            float vmin = w[0];
            for (uint32_t k = 1; k < K; k++) {
                if (w[k] < vmin || (w[k] == vmin && idx[k] < idx[imin])) {
                    vmin = w[k]; imin = (int)k;
                }
            }
            if (v > vmin || (v == vmin && (int32_t)e > idx[imin])) {
                w[imin] = v; idx[imin] = (int32_t)e;
            }
        }
        // Note: mx.argpartition does NOT sort; ordering of the top-K is
        // implementation-defined. Match MLX's "argpartition(kth=-K)[-K:]"
        // ordering as best we can — but for renorm + downstream gather +
        // weighted sum, the order is irrelevant (commutative).
        if (renorm) {
            float ssum = 0.0f;
            for (uint32_t k = 0; k < K; k++) ssum += w[k];
            float r = 1.0f / round_bf16(ssum);
            for (uint32_t k = 0; k < K; k++) w[k] = round_bf16(w[k]*r);
        }
        free(p);
    }
}

void expert_mix_bf16(const bf16* per_expert_out, const float* topk_w,
                     bf16* out, uint32_t L, uint32_t K_top, uint32_t H)
{
    bf16* products = malloc((size_t)K_top*sizeof(*products));
    if (!products) { fprintf(stderr, "expert mixture allocation failed\n"); exit(1); }
    for (uint32_t l = 0; l < L; l++) {
        for (uint32_t h = 0; h < H; h++) {
            for (uint32_t kt = 0; kt < K_top; kt++) {
                float w = topk_w[(size_t)l * K_top + kt];
                float v = bf16_to_f32(per_expert_out[((size_t)l * K_top + kt) * H + h]);
                products[kt] = f32_to_bf16(w*v);
            }
            out[(size_t)l * H + h] = f32_to_bf16(sum_strided_bf16(products, K_top));
        }
    }
    free(products);
}

// ---------- depthwise causal Conv1d ----------------------------------------

void conv1d_depthwise_causal_bf16(const bf16* X, bf16* state,
                                  const bf16* W, bf16* Y,
                                  uint32_t L, uint32_t C, uint32_t K)
{
    // Effective input is concat(state, X) along the time axis, length (K-1)+L.
    // Y[l, c] = sum_{j=0..K-1} W[c, j] * input[l + j, c]
    // After computing Y, the last K-1 rows of input become the new state.
    uint32_t P = K - 1;
    // We need a temporary buffer to read the unified input row-by-row.
    // Allocate (P + L) * C floats — but that's wasteful. Instead read
    // values lazily from state vs X.
    for (uint32_t l = 0; l < L; l++) {
        for (uint32_t c = 0; c < C; c++) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < K; j++) {
                int t = (int)l + (int)j - (int)P;       // input row index
                float v;
                if (t < 0) {
                    v = bf16_to_f32(state[(size_t)(t + P) * C + c]);
                } else {
                    v = bf16_to_f32(X[(size_t)t * C + c]);
                }
                acc += bf16_to_f32(W[(size_t)c * K + j]) * v;
            }
            Y[(size_t)l * C + c] = f32_to_bf16(acc);
        }
    }
    // Update state: last P rows of concat(state, X).
    if (L >= P) {
        memcpy(state, X + (size_t)(L - P) * C, (size_t)P * C * sizeof(bf16));
    } else {
        // First L rows of new state come from the tail of the old state, then X.
        memmove(state, state + (size_t)L * C, (size_t)(P - L) * C * sizeof(bf16));
        memcpy(state + (size_t)(P - L) * C, X, (size_t)L * C * sizeof(bf16));
    }
}

// ---------- GatedDeltaNet ---------------------------------------------------

void compute_g_f32(const bf16* a, const bf16* A_log, const bf16* dt_bias,
                   float* g, uint32_t L, uint32_t Hv)
{
    for (uint32_t l = 0; l < L; l++) {
        for (uint32_t h = 0; h < Hv; h++) {
            float av  = bf16_to_f32(a[(size_t)l * Hv + h]);
            float dt  = bf16_to_f32(dt_bias[h]);
            float alg = bf16_to_f32(A_log[h]);
            float sum = bf16_to_f32(f32_to_bf16(av + dt));
            float e = round_bf16(expf(-fabsf(sum)));
            float sp = round_bf16(fmaxf(sum, 0.0f) + round_bf16(log1pf(e)));
            float r   = expf(-expf(alg) * sp);
            g[(size_t)l * Hv + h] = r;
        }
    }
}

void gated_delta_step_bf16(const bf16* q, const bf16* k, const bf16* v,
                           const float* g, const bf16* beta,
                           float* state,
                           bf16* y,
                           uint32_t Hk, uint32_t Hv,
                           uint32_t Dk, uint32_t Dv)
{
    uint32_t rep = Hv / Hk;
    // state shape: [Hv, Dv, Dk] (f32)
    // For each v_head h_v, its corresponding k_head is h_k = h_v / rep.
    for (uint32_t hv = 0; hv < Hv; hv++) {
        uint32_t hk = hv / rep;
        const bf16* q_row = q + (size_t)hk * Dk;
        const bf16* k_row = k + (size_t)hk * Dk;
        const bf16* v_row = v + (size_t)hv * Dv;
        float gv = g[hv];
        float bv = bf16_to_f32(beta[hv]);
        float* st = state + (size_t)hv * Dv * Dk;
        // 1) state *= g (broadcast over Dv, Dk)
        // 2) kv_mem[dv] = Σ_dk state[dv, dk] * k[dk]
        // 3) delta[dv]  = (v[dv] - kv_mem[dv]) * beta
        // 4) state[dv, dk] += k[dk] * delta[dv]
        // 5) y[dv]      = Σ_dk state[dv, dk] * q[dk]
        // Apply gain first, then merge (1)/(2)/(4)/(5).
        for (uint32_t dv = 0; dv < Dv; dv++) {
            float kv_mem = 0.0f;
            for (uint32_t dk = 0; dk < Dk; dk++) {
                float s = st[(size_t)dv * Dk + dk] * gv;
                st[(size_t)dv * Dk + dk] = s;
                kv_mem += s * bf16_to_f32(k_row[dk]);
            }
            float vv    = bf16_to_f32(v_row[dv]);
            float delta = (vv - kv_mem) * bv;
            float out = 0.0f;
            for (uint32_t dk = 0; dk < Dk; dk++) {
                float s = st[(size_t)dv * Dk + dk] + bf16_to_f32(k_row[dk]) * delta;
                st[(size_t)dv * Dk + dk] = s;
                out += s * bf16_to_f32(q_row[dk]);
            }
            y[(size_t)hv * Dv + dv] = f32_to_bf16(out);
        }
    }
}
