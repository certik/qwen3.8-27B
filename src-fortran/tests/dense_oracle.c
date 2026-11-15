/* Test-only native fixture writer. All expected outputs call the pinned C kernels. */
#include "kernels.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8 && sizeof(bf16) == 2 && sizeof(uint32_t) == 4 &&
               sizeof(float) == 4, "oracle requires 8/16/32-bit native storage");

static FILE *output;
static uint32_t records, comparisons;
static uint32_t compensated_witnesses, weighted_witnesses, scalar_witnesses;
static uint32_t decay_witnesses, mask_witnesses, rope_witnesses;

static void put(const void *data, size_t width, size_t count) {
    if (fwrite(data, width, count, output) != count) {
        fprintf(stderr, "cannot write dense C oracle\n");
        exit(1);
    }
}

static void *array(size_t count, size_t width) {
    if (count > SIZE_MAX / width) exit(1);
    void *p = calloc(count, width);
    if (!p) {
        fprintf(stderr, "cannot allocate dense C oracle fixture\n");
        exit(1);
    }
    return p;
}

static void record(uint32_t op, uint32_t d, uint32_t m, uint32_t n, uint32_t flag) {
    const uint32_t header[] = {op, d, m, n, flag};
    put(header, sizeof(uint32_t), 5);
    records++;
}

static void expected(const bf16 *y, size_t count) {
    put(y, sizeof(bf16), count);
    if (count > UINT32_MAX - comparisons) exit(1);
    comparisons += (uint32_t)count;
}

static void expected_f32(const float *y, size_t count) {
    put(y, sizeof(*y), count);
    if (count > UINT32_MAX - comparisons) exit(1);
    comparisons += (uint32_t)count;
}

static float sample(size_t i, uint32_t salt) {
    return ((int)((i * 37 + salt * 11) % 1021) - 510) / 137.0f;
}

static void weights(uint32_t *w, bf16 *s, bf16 *b, uint32_t k, uint32_t n) {
    for (size_t i = 0; i < (size_t)n * k / 4; i++) {
        uint32_t packed = 0;
        for (uint32_t byte = 0; byte < 4; byte++)
            packed |= (uint32_t)((i * 29 + byte * 83) % 256) << (byte * 8);
        w[i] = packed;
    }
    for (size_t i = 0; i < (size_t)n * k / 64; i++) {
        s[i] = f32_to_bf16(sample(i, 3) / 64.0f);
        b[i] = f32_to_bf16(sample(i, 7) / 2.0f);
    }
}

static void embedding(uint32_t d) {
    const uint32_t m = 7, v = 5;
    const int32_t ids[] = {0, 4, 1, 4, 2, 0, 3};
    uint32_t *w = array((size_t)v * d / 4, sizeof(*w));
    bf16 *s = array((size_t)v * d / 64, sizeof(*s));
    bf16 *b = array((size_t)v * d / 64, sizeof(*b));
    bf16 *y = array((size_t)m * d, sizeof(*y));
    weights(w, s, b, d, v);
    embed_dequant_gather_q8(ids, w, s, b, y, m, d);
    record(1, d, m, v, 0);
    put(w, sizeof(*w), (size_t)v * d / 4);
    put(s, sizeof(*s), (size_t)v * d / 64);
    put(b, sizeof(*b), (size_t)v * d / 64);
    put(ids, sizeof(*ids), m);
    expected(y, (size_t)m * d);
    fprintf(stderr, "embedding: D=%u M=%u V=%u\n", d, m, v);
    free(w); free(s); free(b); free(y);
}

static void linear(uint32_t k, uint32_t m, uint32_t n, uint32_t has_bias) {
    bf16 *x = array((size_t)m * k, sizeof(*x));
    uint32_t *w = array((size_t)n * k / 4, sizeof(*w));
    bf16 *s = array((size_t)n * k / 64, sizeof(*s));
    bf16 *b = array((size_t)n * k / 64, sizeof(*b));
    bf16 *bias = array(n, sizeof(*bias));
    bf16 *y = array((size_t)m * n, sizeof(*y));
    weights(w, s, b, k, n);
    for (size_t i = 0; i < (size_t)m * k; i++) x[i] = f32_to_bf16(sample(i, 13));
    for (uint32_t i = 0; i < n; i++) bias[i] = f32_to_bf16(sample(i, 17) * 8.0f);
    linear_q8_bf16(x, w, s, b, has_bias ? bias : NULL, y, m, k, n);
    record(2, k, m, n, has_bias);
    put(x, sizeof(*x), (size_t)m * k);
    put(w, sizeof(*w), (size_t)n * k / 4);
    put(s, sizeof(*s), (size_t)n * k / 64);
    put(b, sizeof(*b), (size_t)n * k / 64);
    if (has_bias) put(bias, sizeof(*bias), n);
    expected(y, (size_t)m * n);
    fprintf(stderr, "linear: K=%u M=%u N=%u bias=%u\n", k, m, n, has_bias);
    free(x); free(w); free(s); free(b); free(bias); free(y);
}

static void norm(uint32_t d, uint32_t m) {
    const float eps = 1.0e-6f, scalar = 0.1234567f;
    bf16 *x = array((size_t)d * m, sizeof(*x));
    bf16 *w = array(d, sizeof(*w));
    bf16 *y = array((size_t)d * m, sizeof(*y));
    for (size_t i = 0; i < (size_t)d * m; i++)
        x[i] = f32_to_bf16(ldexpf(sample(i, 23), (int)(i % 13) - 6));
    if (m > 1) memset(x + d, 0, d * sizeof(*x));
    for (uint32_t i = 0; i < d; i++)
        w[i] = f32_to_bf16(ldexpf(sample(i, 29), (int)(i % 9) - 3));
    for (uint32_t weighted = 0; weighted < 2; weighted++) {
        rmsnorm_bf16(x, weighted ? w : NULL, y, m, d, eps);
        record(3, d, m, 0, weighted);
        put(x, sizeof(*x), (size_t)d * m);
        put(&eps, sizeof(eps), 1);
        if (weighted) put(w, sizeof(*w), d);
        expected(y, (size_t)d * m);
        for (uint32_t row = 0; row < m; row++) {
            float sq = 0.0f, correction = 0.0f, naive = 0.0f;
            for (uint32_t i = 0; i < d; i++) {
                float v = bf16_to_f32(x[(size_t)row * d + i]);
                float term = v * v - correction;
                float next = sq + term;
                correction = (next - sq) - term;
                sq = next;
                naive += v * v;
            }
            float rrms = 1.0f / sqrtf(sq / (float)d + eps);
            float naive_rrms = 1.0f / sqrtf(naive / (float)d + eps);
            for (uint32_t i = 0; i < d; i++) {
                size_t at = (size_t)row * d + i;
                float v = bf16_to_f32(x[at]);
                float nv = v * naive_rrms;
                if (weighted) nv = bf16_to_f32(f32_to_bf16(nv)) * bf16_to_f32(w[i]);
                compensated_witnesses += f32_to_bf16(nv) != y[at];
                if (weighted)
                    weighted_witnesses += f32_to_bf16(v * rrms * bf16_to_f32(w[i])) != y[at];
            }
        }
    }
    rmsnorm_scale_bf16(x, y, m, d, eps, scalar);
    record(4, d, m, 0, 0);
    put(x, sizeof(*x), (size_t)d * m);
    put(&eps, sizeof(eps), 1);
    put(&scalar, sizeof(scalar), 1);
    expected(y, (size_t)d * m);
    bf16 *normalized = array((size_t)d * m, sizeof(*normalized));
    rmsnorm_bf16(x, NULL, normalized, m, d, eps);
    for (size_t i = 0; i < (size_t)d * m; i++)
        scalar_witnesses += f32_to_bf16(bf16_to_f32(normalized[i]) * scalar) != y[i];
    fprintf(stderr, "norm: D=%u M=%u (unweighted, weighted, scale)\n", d, m);
    free(x); free(w); free(y); free(normalized);
}

static void elements(void) {
    const uint32_t n = 4099;
    const float special[] = {0.0f, -0.0f, 0.5f, -0.5f, 1.0078125f, -1.0078125f,
                             88.0f, -88.0f, 1000.0f, -1000.0f};
    bf16 *gate = array(n, sizeof(*gate));
    bf16 *up = array(n, sizeof(*up));
    bf16 *y = array((size_t)6 * n, sizeof(*y));
    for (uint32_t i = 0; i < n; i++) {
        gate[i] = (bf16)((i * 73 % 0x4300) | ((i % 2) << 15));
        if (i < sizeof(special) / sizeof(special[0])) gate[i] = f32_to_bf16(special[i]);
        up[i] = f32_to_bf16(sample(i, 31));
        if (i % 5 == 0) up[i] = f32_to_bf16(-bf16_to_f32(gate[i]));
    }
    silu_mul_bf16(gate, up, y, n);
    silu_mul_f32_gate_bf16(gate, up, y + n, n);
    elemwise_sigmoid_mul_bf16(gate, up, y + 2 * n, n);
    memcpy(y + 3 * n, gate, n * sizeof(*gate));
    residual_add_bf16(y + 3 * n, up, n);
    memcpy(y + 4 * n, gate, n * sizeof(*gate));
    silu_inplace_bf16(y + 4 * n, n);
    memcpy(y + 5 * n, gate, n * sizeof(*gate));
    sigmoid_inplace_bf16(y + 5 * n, n);
    if (memcmp(y, y + n, n * sizeof(*y)) == 0) {
        fprintf(stderr, "element corpus must distinguish the two SiLU variants\n");
        exit(1);
    }
    record(5, n, 1, 0, 0);
    put(gate, sizeof(*gate), n);
    put(up, sizeof(*up), n);
    expected(y, (size_t)6 * n);
    fprintf(stderr, "elements: N=%u (6 operations, distinct SiLU outputs)\n", n);
    free(gate); free(up); free(y);
}

static void argmax_case(uint32_t n, uint32_t mode) {
    bf16 *x = array(n, sizeof(*x));
    for (uint32_t i = 0; i < n; i++) x[i] = f32_to_bf16(-fabsf(sample(i, 37)) - 1.0f);
    if (mode == 0) {
        x[0] = f32_to_bf16(-0.0f);
        x[n - 1] = f32_to_bf16(0.0f);
    } else if (mode == 1) {
        x[2] = x[n - 1] = f32_to_bf16(7.5f);
    } else if (mode == 2) {
        x[n - 1] = f32_to_bf16(7.5f);
    }
    int32_t index;
    argmax_bf16(x, &index, n);
    record(6, n, 1, 0, 0);
    put(x, sizeof(*x), n);
    put(&index, sizeof(index), 1);
    comparisons++;
    fprintf(stderr, "argmax: N=%u mode=%u index=%d\n", n, mode, index);
    free(x);
}

static void decay(uint32_t hv, uint32_t length, uint32_t mode) {
    size_t count = (size_t)hv * length;
    bf16 *a = array(count, sizeof(*a));
    bf16 *alg = array(hv, sizeof(*alg));
    bf16 *dt = array(hv, sizeof(*dt));
    float *g = array(count, sizeof(*g));
    for (uint32_t h = 0; h < hv; h++) {
        alg[h] = f32_to_bf16(-0.5f - (float)(h % 9) / 4.0f);
        dt[h] = mode == 0 ? 0 : f32_to_bf16(sample(h, 41) / 2.0f);
        if (mode == 2) alg[h] = dt[h];
    }
    for (size_t i = 0; i < count; i++) {
        /* All 65,280 finite BF16 images, including both signed zeros. */
        a[i] = mode == 0 ? (bf16)((i % 0x7f80) | ((i / 0x7f80) << 15)) :
                           f32_to_bf16(sample(i, 43) * (i % 3 == 0 ? 32.0f : 1.0f));
    }
    compute_g_f32(a, alg, mode == 2 ? alg : dt, g, length, hv);
    record(7, hv, length, 0, mode == 2);
    put(a, sizeof(*a), count);
    put(alg, sizeof(*alg), hv);
    put(dt, sizeof(*dt), hv);
    expected_f32(g, count);
    for (size_t i = 0; i < count; i++)
        decay_witnesses += bf16_to_f32(f32_to_bf16(g[i])) != g[i];
    fprintf(stderr, "compute_g: Hv=%u L=%u mode=%u\n", hv, length, mode);
    free(a); free(alg); free(dt); free(g);
}

static void convolution(uint32_t channels, uint32_t kernel) {
    const uint32_t calls = 6;
    const uint32_t lengths[] = {1, 2, 1, kernel + 1, 2, 1};
    size_t carry = (size_t)(kernel - 1) * channels;
    bf16 *w = array((size_t)kernel * channels, sizeof(*w));
    /* A non-null pointer is also used for the zero-length K=1 carry. */
    bf16 *state = array(carry ? carry : 1, sizeof(*state));
    for (size_t i = 0; i < (size_t)kernel * channels; i++)
        w[i] = f32_to_bf16(sample(i, 47) / 4.0f);
    for (size_t i = 0; i < carry; i++) state[i] = f32_to_bf16(sample(i, 53));
    record(8, channels, calls, kernel, 0);
    put(w, sizeof(*w), (size_t)kernel * channels);
    put(state, sizeof(*state), carry);
    for (uint32_t step = 0; step < calls; step++) {
        uint32_t meta[] = {lengths[step], step == 2};
        size_t count = (size_t)lengths[step] * channels;
        bf16 *x = array(count, sizeof(*x)), *y = array(count, sizeof(*y));
        for (size_t i = 0; i < count; i++) x[i] = f32_to_bf16(sample(i, 59 + step * 7));
        if (meta[1]) memset(state, 0, carry * sizeof(*state));
        conv1d_depthwise_causal_bf16(x, state, w, y, lengths[step], channels, kernel);
        put(meta, sizeof(*meta), 2);
        put(x, sizeof(*x), count);
        expected(y, count);
        expected(state, carry);
        free(x); free(y);
    }
    fprintf(stderr, "conv: C=%u K=%u, six calls (L=1,2,1,K+1,2,1), reset at call 3\n",
            channels, kernel);
    free(w); free(state);
}

static void rope(uint32_t d, uint32_t length, uint32_t heads, uint32_t rot, uint32_t offset) {
    size_t count = (size_t)d * length * heads;
    bf16 *x = array(count, sizeof(*x)), *y = array(count, sizeof(*y));
    float *freq = array(rot ? rot / 2 : 1, sizeof(*freq));
    for (size_t i = 0; i < count; i++) x[i] = f32_to_bf16(sample(i, 61));
    for (uint32_t i = 0; i < rot / 2; i++) freq[i] = 1.0f / (1.0f + (float)(i * i + i));
    memcpy(y, x, count * sizeof(*x));
    rope_partial_inplace_bf16(y, freq, length, heads, d, rot, offset);
    for (size_t i = 0; i < count; i++) rope_witnesses += x[i] != y[i];
    record(9, d, length, heads, 0);
    const uint64_t position = offset;
    put(&rot, sizeof(rot), 1);
    put(&position, sizeof(position), 1);
    put(freq, sizeof(*freq), rot / 2);
    put(x, sizeof(*x), count);
    expected(y, count);
    comparisons += (uint32_t)count; /* Full and chunked explicit in-place calls. */
    fprintf(stderr, "rope: D=%u L=%u H=%u rot=%u offset=%u\n", d, length, heads, rot, offset);
    free(x); free(y); free(freq);
}

static void attention(uint32_t d, uint32_t lq, uint32_t lk, uint32_t nq, uint32_t nkv,
                      uint32_t offset, uint32_t causal) {
    size_t qcount = (size_t)d * lq * nq, kvcount = (size_t)d * lk * nkv;
    bf16 *q = array(qcount, sizeof(*q)), *k = array(kvcount, sizeof(*k));
    bf16 *v = array(kvcount, sizeof(*v)), *y = array(qcount, sizeof(*y));
    bf16 *other_mask = array(qcount, sizeof(*other_mask));
    const float scale = 0.7312345f / sqrtf((float)d);
    for (size_t i = 0; i < qcount; i++)
        q[i] = f32_to_bf16(sample(i + (size_t)offset * nq * d, 67) / 2.0f);
    for (size_t i = 0; i < kvcount; i++) {
        k[i] = f32_to_bf16(sample(i, 71) / 2.0f);
        v[i] = f32_to_bf16(sample(i, 73));
    }
    sdpa_gqa_bf16(q, k, v, y, lq, lk, nq, nkv, d, offset, scale, (int)causal);
    sdpa_gqa_bf16(q, k, v, other_mask, lq, lk, nq, nkv, d, offset, scale, !causal);
    for (size_t i = 0; i < qcount; i++) mask_witnesses += y[i] != other_mask[i];
    record(10, d, lq, lk, causal);
    const uint32_t heads[] = {nq, nkv};
    const uint64_t position = offset;
    put(heads, sizeof(*heads), 2);
    put(&position, sizeof(position), 1);
    put(&scale, sizeof(scale), 1);
    put(q, sizeof(*q), qcount);
    put(k, sizeof(*k), kvcount);
    put(v, sizeof(*v), kvcount);
    expected(y, qcount);
    fprintf(stderr, "sdpa: D=%u Lq=%u Lk=%u Nq/Nkv=%u/%u offset=%u causal=%u\n",
            d, lq, lk, nq, nkv, offset, causal);
    free(q); free(k); free(v); free(y); free(other_mask);
}

static void recurrent(uint32_t dk, uint32_t dv, uint32_t hk, uint32_t hv) {
    const uint32_t calls = 6;
    size_t qcount = (size_t)dk * hk, vcount = (size_t)dv * hv, scount = (size_t)dk * vcount;
    bf16 *q = array(qcount, sizeof(*q)), *k = array(qcount, sizeof(*k));
    bf16 *v = array(vcount, sizeof(*v)), *beta = array(hv, sizeof(*beta));
    bf16 *y = array(vcount, sizeof(*y));
    float *g = array(hv, sizeof(*g)), *state = array(scount, sizeof(*state));
    for (size_t i = 0; i < scount; i++) state[i] = sample(i, 79) / 64.0f;
    record(11, dk, dv, hk, 0);
    const uint32_t meta[] = {hv, calls};
    put(meta, sizeof(*meta), 2);
    put(state, sizeof(*state), scount);
    for (uint32_t step = 0; step < calls; step++) {
        const uint32_t reset = step == 3;
        for (size_t i = 0; i < qcount; i++) {
            q[i] = f32_to_bf16(sample(i, 83 + step * 3) / 16.0f);
            k[i] = step == 1 ? q[i] : f32_to_bf16(sample(i, 89 + step * 5) / 16.0f);
        }
        for (size_t i = 0; i < vcount; i++) v[i] = f32_to_bf16(sample(i, 97 + step * 7));
        for (uint32_t h = 0; h < hv; h++) {
            beta[h] = f32_to_bf16((float)((h + step) % 9) / 8.0f);
            g[h] = (h + step) % 7 == 0 ? 0.0f :
                   (h + step) % 7 == 1 ? 1.0f : 0.6172839f + (float)(h % 5) / 16.0f;
        }
        if (reset) memset(state, 0, scount * sizeof(*state));
        gated_delta_step_bf16(q, step == 1 ? q : k, v, g, beta, state, y, hk, hv, dk, dv);
        put(&reset, sizeof(reset), 1);
        put(q, sizeof(*q), qcount);
        put(k, sizeof(*k), qcount);
        put(v, sizeof(*v), vcount);
        put(g, sizeof(*g), hv);
        put(beta, sizeof(*beta), hv);
        expected(y, vcount);
        expected_f32(state, scount);
    }
    fprintf(stderr, "gated_delta: Dk/Dv=%u/%u Hk/Hv=%u/%u, six steps, reset at step 4, q=k at step 2\n",
            dk, dv, hk, hv);
    free(q); free(k); free(v); free(beta); free(y); free(g); free(state);
}

int main(int argc, char **argv) {
    if (argc != 2 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) {
        fprintf(stderr, "usage: dense_oracle native-output.bin (IEEE binary32 required)\n");
        return 1;
    }
    output = fopen(argv[1], "wb");
    if (!output) return 1;
    const uint32_t header[] = {0x464b3031, 0x01020304, 0x3f800000};
    const bf16 marker = 0x8123;
    const float real_marker = 1.25f;
    put(header, sizeof(*header), 3);
    put(&marker, sizeof(marker), 1);
    put(&real_marker, sizeof(real_marker), 1);
    embedding(64); embedding(192);
    linear(64, 1, 7, 0); linear(128, 2, 5, 1);
    linear(128, 17, 3, 0); linear(128, 18, 3, 0);
    linear(64, 18, 3, 1);
    linear(320, 18, 3, 1); linear(512, 18, 3, 0);
    linear(576, 18, 3, 0); linear(1984, 18, 3, 1);
    linear(2048, 18, 3, 0);
    linear(2112, 11, 3, 0); linear(2112, 12, 3, 1);
    linear(4096, 12, 3, 0);
    linear(4160, 9, 3, 0); linear(4160, 10, 3, 1);
    linear(32768, 10, 3, 1);
    linear(5184, 65, 65, 1); /* divisor search: 56 -> 27 parts */
    linear(64, 11, 2049, 0); linear(64, 12, 2049, 0);
    linear(64, 9, 4097, 0); linear(64, 10, 4097, 1);
    linear(64, 481, 993, 0); /* 512 tiles: rounded weights, no split */
    norm(1, 3); norm(7, 3); norm(63, 3); norm(64, 3);
    norm(65, 3); norm(127, 3); norm(5120, 17);
    elements();
    argmax_case(1, 0); argmax_case(17, 0); argmax_case(17, 1);
    argmax_case(513, 2); argmax_case(65, 3);
    decay(1, 1, 1); decay(48, 1360, 0); decay(7, 19, 1); decay(3, 5, 2);
    convolution(1, 1); convolution(5, 2); convolution(7, 4); convolution(3, 7);
    rope(7, 5, 3, 4, 2); rope(8, 3, 2, 8, 0); rope(5, 2, 1, 0, 7);
    rope(256, 4, 3, 64, 16777219); rope(9, 2, 2, 6, UINT32_MAX - 1);
    attention(7, 3, 9, 6, 2, 2, 1);
    attention(7, 3, 9, 6, 2, 2, 0);
    attention(64, 1, 9, 6, 2, 6, 1);
    attention(96, 8, 11, 6, 2, 0, 1);
    attention(128, 8, 13, 8, 2, 3, 1);
    attention(128, 8, 13, 10, 2, 3, 1); /* group*Lq=40: fallback */
    attention(256, 8, 11, 6, 2, 2, 1);
    attention(64, 9, 13, 6, 2, 2, 1);
    attention(80, 9, 13, 6, 2, 2, 0);
    attention(128, 9, 13, 6, 2, 2, 1);
    attention(96, 9, 13, 6, 2, 2, 1);
    attention(256, 9, 13, 6, 2, 2, 1);
    attention(80, 8, 13, 6, 2, 2, 1);
    attention(256, 1, 3, 33, 1, 0, 1); /* group*Lq=33: fallback */
    attention(7, 1, 1, 1, 1, 0, 1);
    /* Same sequence/inputs, successively visible cache prefixes. */
    attention(256, 4, 4, 3, 1, 0, 1);
    attention(256, 1, 5, 3, 1, 4, 1);
    attention(256, 2, 7, 3, 1, 5, 1);
    recurrent(1, 1, 1, 1); recurrent(7, 5, 16, 48);
    recurrent(128, 128, 1, 3); recurrent(9, 3, 2, 2);
    if (!compensated_witnesses || !weighted_witnesses || !scalar_witnesses) {
        fprintf(stderr, "norm corpus lacks rounding witnesses: compensated=%u weight=%u scalar=%u\n",
                compensated_witnesses, weighted_witnesses, scalar_witnesses);
        return 1;
    }
    if (!decay_witnesses || !mask_witnesses || !rope_witnesses) {
        fprintf(stderr, "attention/state corpus lacks precision/mask/rotation witnesses\n");
        return 1;
    }
    const uint32_t footer[] = {0, records, comparisons};
    put(footer, sizeof(*footer), 3);
    if (fclose(output) != 0) return 1;
    fprintf(stderr, "rounding witnesses: compensated=%u weight=%u scalar=%u\n",
            compensated_witnesses, weighted_witnesses, scalar_witnesses);
    fprintf(stderr, "attention/state witnesses: non-BF16-g=%u causal-mask=%u rotated=%u\n",
            decay_witnesses, mask_witnesses, rope_witnesses);
    fprintf(stderr, "PASS: C oracle wrote %u records, %u exact output comparisons\n", records, comparisons);
    return 0;
}
