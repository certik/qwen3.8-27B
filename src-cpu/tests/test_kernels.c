// test_kernels.c — per-kernel correctness checks against MLX oracle dumps.
//
// Run with `./test_kernels --model PATH [name ...]` to validate one or
// more named kernels. With no names, runs the full suite.

#include "../kernels.h"
#include "../safetensors.h"
#include "../utils/bf16.h"
#include "load_ref.h"
#include "compare.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------- helpers --------------------------------------------------------

static st_archive* g_arch = NULL;

static ref_t* optional_ref(const char* name) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", ref_directory ? ref_directory : "src-cpu/refs", name);
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fclose(f);
    return ref_load(name);
}

static bf16* from_ref(const ref_t* r) {
    if (r->dtype != 0) { fprintf(stderr, "expected floating-point fixture\n"); exit(1); }
    bf16* data = calloc(r->n_elem, sizeof(*data));
    if (!data) exit(1);
    for (size_t i = 0; i < r->n_elem; i++) data[i] = f32_to_bf16(((float*)r->data)[i]);
    return data;
}

static const void* tens(const char* name, st_dtype want, size_t* nelem) {
    const st_tensor* t = st_find(g_arch, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (t->dtype != want) {
        fprintf(stderr, "tensor %s dtype mismatch: want %s got %s\n",
                name, st_dtype_name(want), st_dtype_name(t->dtype));
        exit(1);
    }
    if (nelem) {
        size_t n = 1;
        for (int i = 0; i < t->rank; i++) n *= (size_t)t->shape[i];
        *nelem = n;
    }
    return t->data;
}

// ---------- individual tests ----------------------------------------------

static int test_embed_gather(void) {
    printf("[embed_gather_q8]\n");
    ref_t* ids_r = ref_load("src-cpu/refs/input_ids.bin");
    ref_t* x_r   = ref_load("src-cpu/refs/x_after_embed.bin");
    size_t Lp = ids_r->n_elem;
    size_t D = (size_t)x_r->shape[1];

    size_t E_n;
    const uint32_t* E_W = tens("language_model.model.embed_tokens.weight", ST_DTYPE_U32, &E_n);
    const bf16* E_S = tens("language_model.model.embed_tokens.scales", ST_DTYPE_BF16, NULL);
    const bf16* E_B = tens("language_model.model.embed_tokens.biases", ST_DTYPE_BF16, NULL);

    bf16* Y = calloc(Lp * D, sizeof(bf16));
    embed_dequant_gather_q8((const int32_t*)ids_r->data, E_W, E_S, E_B,
                            Y, (uint32_t)Lp, (uint32_t)D);

    double d = err_check(Y, (const float*)x_r->data, Lp * D, 0.005, "embed_dequant_gather_q8");
    free(Y); ref_free(ids_r); ref_free(x_r);
    return d;
}

static int test_rmsnorm_input0(void) {
    printf("[rmsnorm (input_layernorm layer 0)]\n");
    ref_t* xin  = ref_load("src-cpu/refs/x_in_0.bin");
    ref_t* hout = ref_load("src-cpu/refs/h_after_input_norm_0.bin");
    size_t L = (size_t)xin->shape[0], D = (size_t)xin->shape[1];

    const bf16* W = tens("language_model.model.layers.0.input_layernorm.weight",
                        ST_DTYPE_BF16, NULL);

    // Convert f32 input to bf16 (the C kernel expects bf16).
    bf16* X = calloc(L * D, sizeof(bf16));
    for (size_t i = 0; i < L * D; i++) X[i] = f32_to_bf16(((const float*)xin->data)[i]);
    bf16* Y = calloc(L * D, sizeof(bf16));

    rmsnorm_bf16(X, W, Y, (uint32_t)L, (uint32_t)D, 1e-6f);

    double d = err_check(Y, (const float*)hout->data, L * D, 0.01,
                         "rmsnorm_bf16 (layer 0 input)");
    free(X); free(Y); ref_free(xin); ref_free(hout);
    return d;
}

static int test_linear_q8_qproj_l3(void) {
    printf("[linear_q8 (q_proj layer 3)]\n");
    ref_t* h_r = ref_load("src-cpu/refs/h_after_input_norm_3.bin");
    ref_t* q_r = ref_load("src-cpu/refs/q_proj_3.bin");
    size_t L = (size_t)h_r->shape[0], K = (size_t)h_r->shape[1];
    size_t N = (size_t)q_r->shape[1];

    const uint32_t* W = tens("language_model.model.layers.3.self_attn.q_proj.weight",
                              ST_DTYPE_U32, NULL);
    const bf16* S = tens("language_model.model.layers.3.self_attn.q_proj.scales",
                          ST_DTYPE_BF16, NULL);
    const bf16* B = tens("language_model.model.layers.3.self_attn.q_proj.biases",
                          ST_DTYPE_BF16, NULL);

    bf16* X = calloc(L * K, sizeof(bf16));
    for (size_t i = 0; i < L * K; i++) X[i] = f32_to_bf16(((const float*)h_r->data)[i]);
    bf16* Y = calloc(L * N, sizeof(bf16));

    linear_q8_bf16(X, W, S, B, NULL, Y, (uint32_t)L, (uint32_t)K, (uint32_t)N);

    double d = err_check(Y, (const float*)q_r->data, L * N, 0.03,
                         "linear_q8_bf16 (q_proj L3)");
    free(X); free(Y); ref_free(h_r); ref_free(q_r);
    return d;
}

static int test_silu_mul(void) {
    printf("[silu_mul (SwiGLU mid; expert layer 0)]\n");
    ref_t* gp = ref_load("src-cpu/refs/gate_proj_0.bin");
    ref_t* up = ref_load("src-cpu/refs/up_proj_0.bin");
    ref_t* mid = ref_load("src-cpu/refs/swiglu_mid_0.bin");
    size_t N = gp->n_elem;
    bf16 *G = calloc(N, sizeof(bf16)), *U = calloc(N, sizeof(bf16)), *Y = calloc(N, sizeof(bf16));
    for (size_t i = 0; i < N; i++) {
        G[i] = f32_to_bf16(((const float*)gp->data)[i]);
        U[i] = f32_to_bf16(((const float*)up->data)[i]);
    }
    silu_mul_bf16(G, U, Y, (uint32_t)N);
    double d = err_check(Y, (const float*)mid->data, N, 0.005, "silu_mul_bf16");
    free(G); free(U); free(Y); ref_free(gp); ref_free(up); ref_free(mid);
    return d;
}

static int test_compute_g(void) {
    printf("[compute_g (layer 0)]\n");
    ref_t* ar = ref_load("src-cpu/refs/a_0.bin");
    ref_t* gr = ref_load("src-cpu/refs/g_0.bin");
    size_t L = (size_t)ar->shape[0], Hv = (size_t)ar->shape[1];
    const bf16* Al = tens("language_model.model.layers.0.linear_attn.A_log",
                            ST_DTYPE_BF16, NULL);
    const bf16* dt = tens("language_model.model.layers.0.linear_attn.dt_bias",
                            ST_DTYPE_BF16, NULL);

    bf16* A = calloc(L * Hv, sizeof(bf16));
    for (size_t i = 0; i < L * Hv; i++) A[i] = f32_to_bf16(((const float*)ar->data)[i]);
    float* Y = calloc(L * Hv, sizeof(float));
    compute_g_f32(A, Al, dt, Y, (uint32_t)L, (uint32_t)Hv);

    int d = check_f32(Y, gr->data, L * Hv, 1e-5, 1e-5, "compute_g_f32");
    if (d) {
        int shown = 0;
        for (size_t i = 0; i < L*Hv && shown < 5; i++) {
            float want = ((float*)gr->data)[i];
            if (fabsf(Y[i]-want) > 1e-5f + 1e-5f*fabsf(want)) {
                printf("    g[%zu] a=%a dt=%a A_log=%a got=%a want=%a\n",
                       i, bf16_to_f32(A[i]), bf16_to_f32(dt[i%Hv]),
                       bf16_to_f32(Al[i%Hv]), Y[i], want);
                shown++;
            }
        }
    }
    free(A); free(Y); ref_free(ar); ref_free(gr);
    return d;
}

static int test_recurrent_norms(void) {
    ref_t *qr = ref_load("q_lin_0.bin"), *kr = ref_load("k_lin_0.bin");
    ref_t *qn = ref_load("q_norm_lin_0.bin"), *kn = ref_load("k_norm_lin_0.bin");
    bf16 *q = from_ref(qr), *k = from_ref(kr), *out = calloc(qr->n_elem, sizeof(*out));
    uint32_t d = (uint32_t)qr->shape[2], rows = (uint32_t)(qr->shape[0]*qr->shape[1]);
    float scale = 1.0f/sqrtf((float)d);
    rmsnorm_scale_bf16(q, out, rows, d, 1e-6f, scale*scale);
    int fail = err_check(out, qn->data, qr->n_elem, .01, "recurrent q RMSNorm/scale");
    rmsnorm_scale_bf16(k, out, rows, d, 1e-6f, scale);
    fail |= err_check(out, kn->data, kr->n_elem, .01, "recurrent k RMSNorm/scale");
    free(q); free(k); free(out);
    ref_free(qr); ref_free(kr); ref_free(qn); ref_free(kn);
    ref_t *yr = ref_load("delta_y_0.bin"), *zr = ref_load("z_0.bin");
    ref_t *want = ref_load("norm_out_0.bin");
    bf16 *y = from_ref(yr), *z = from_ref(zr);
    out = calloc(yr->n_elem, sizeof(*out));
    const bf16* w = tens("language_model.model.layers.0.linear_attn.norm.weight", ST_DTYPE_BF16, NULL);
    rmsnorm_bf16(y, w, out, (uint32_t)(yr->shape[0]*yr->shape[1]), (uint32_t)yr->shape[2], 1e-6f);
    silu_mul_f32_gate_bf16(z, out, out, (uint32_t)yr->n_elem);
    fail |= err_check(out, want->data, yr->n_elem, .01, "RMSNormGated FP32 SiLU/product");
    free(y); free(z); free(out);
    ref_free(yr); ref_free(zr); ref_free(want);
    return fail;
}

static int test_dense_ffn(void) {
    ref_t* h = ref_load("h_after_post_norm_0.bin");
    ref_t* gate = ref_load("gate_proj_0.bin");
    ref_t* up = ref_load("up_proj_0.bin");
    ref_t* mid = ref_load("swiglu_mid_0.bin");
    ref_t* down = ref_load("dense_down_0.bin");
    uint32_t l = (uint32_t)h->shape[0], k = (uint32_t)h->shape[1], n = (uint32_t)gate->shape[1];
    const char* names[] = {"gate_proj", "up_proj", "down_proj"};
    ref_t* inputs[] = {h, h, mid};
    ref_t* outputs[] = {gate, up, down};
    int fail = 0;
    for (int j = 0; j < 3; j++) {
        char name[256];
        snprintf(name, sizeof(name), "language_model.model.layers.0.mlp.%s.weight", names[j]);
        const uint32_t* w = tens(name, ST_DTYPE_U32, NULL);
        snprintf(name, sizeof(name), "language_model.model.layers.0.mlp.%s.scales", names[j]);
        const bf16* s = tens(name, ST_DTYPE_BF16, NULL);
        snprintf(name, sizeof(name), "language_model.model.layers.0.mlp.%s.biases", names[j]);
        const bf16* b = tens(name, ST_DTYPE_BF16, NULL);
        bf16* x = from_ref(inputs[j]);
        bf16* y = calloc(outputs[j]->n_elem, sizeof(*y));
        linear_q8_bf16(x, w, s, b, NULL, y, l, j == 2 ? n : k, j == 2 ? k : n);
        fail |= err_check(y, outputs[j]->data, outputs[j]->n_elem, .03, names[j]);
        free(x); free(y);
    }
    ref_free(h); ref_free(gate); ref_free(up); ref_free(mid); ref_free(down);
    return fail;
}

static int test_softmax_topk(void) {
    printf("[softmax_topk (router layer 0)]\n");
    ref_t* logits = ref_load("src-cpu/refs/router_logits_0.bin");
    ref_t* tk_idx = ref_load("src-cpu/refs/topk_idx_0.bin");
    ref_t* tk_w   = ref_load("src-cpu/refs/topk_score_0.bin");
    size_t L = (size_t)logits->shape[0], E = (size_t)logits->shape[1];
    size_t K = (size_t)tk_idx->shape[1];

    bf16* X = calloc(L * E, sizeof(bf16));
    for (size_t i = 0; i < L * E; i++) X[i] = f32_to_bf16(((const float*)logits->data)[i]);
    int32_t* idx = calloc(L * K, sizeof(int32_t));
    float*   w   = calloc(L * K, sizeof(float));
    softmax_topk_bf16(X, idx, w, (uint32_t)L, (uint32_t)E, (uint32_t)K, /*renorm=*/1);

    // Compare top-K SETS (order is implementation-defined).
    int errs = 0;
    const int32_t* want_idx = (const int32_t*)tk_idx->data;
    const float*   want_w   = (const float*)tk_w->data;
    for (size_t l = 0; l < L; l++) {
        for (size_t k = 0; k < K; k++) {
            int e = idx[l * K + k];
            int found = 0; float fw = 0.0f;
            for (size_t kk = 0; kk < K; kk++) {
                if (want_idx[l * K + kk] == e) { found = 1; fw = want_w[l * K + kk]; break; }
            }
            if (!found) {
                printf("    row %zu unexpected expert %d; native=[", l, e);
                for (size_t kk = 0; kk < K; kk++) printf("%s%d", kk ? "," : "", want_idx[l*K+kk]);
                printf("] logits=[");
                for (size_t ee = 0; ee < E; ee++) printf("%s%g", ee ? "," : "", bf16_to_f32(X[l*E+ee]));
                printf("]\n");
                errs++;
            }
            else {
                double d = fabs(fw - w[l * K + k]);
                if (!isfinite(fw) || !isfinite(w[l*K+k]) || d > 5e-3) {
                    printf("    row %zu expert %d: w=%g want=%g\n", l, e, w[l*K+k], fw);
                    errs++;
                }
            }
        }
    }
    printf("  topk_set                          mismatches=%d (LxK=%zu)  %s\n",
           errs, L*K, errs == 0 ? "OK" : "FAIL");
    free(X); free(idx); free(w);
    ref_free(logits); ref_free(tk_idx); ref_free(tk_w);
    return errs == 0 ? 0 : 1;
}

static int test_rope_partial(void) {
    printf("[rope_partial_inplace (layer 3)]\n");
    ref_t* q_pre  = ref_load("src-cpu/refs/q_after_qnorm_3.bin");
    ref_t* q_post = ref_load("src-cpu/refs/q_after_rope_3.bin");
    size_t L = (size_t)q_pre->shape[0], H = (size_t)q_pre->shape[1], D = (size_t)q_pre->shape[2];
    uint32_t D_rot = (uint32_t)(D / 4);     // partial_rotary_factor = 0.25

    bf16* X = calloc(L * H * D, sizeof(bf16));
    for (size_t i = 0; i < L * H * D; i++) X[i] = f32_to_bf16(((const float*)q_pre->data)[i]);

    // freqs[d/2] = 1 / (base ** (d / D_rot))  for d in [0, D_rot/2)
    float base = 1e7f;
    uint32_t Dh = D_rot / 2;
    float* freqs = calloc(Dh, sizeof(float));
    for (uint32_t d = 0; d < Dh; d++) {
        freqs[d] = 1.0f / powf(base, (float)(2*d) / (float)D_rot);
    }
    ref_t* cached = optional_ref("cache_3_0.bin");
    uint32_t offset = cached ? (uint32_t)(cached->shape[2]-(int64_t)L) : 0;
    rope_partial_inplace_bf16(X, freqs, (uint32_t)L, (uint32_t)H, (uint32_t)D,
                              D_rot, offset);
    ref_free(cached);

    double d = err_check(X, (const float*)q_post->data, L * H * D, 0.01,
                         "rope_partial_inplace_bf16 (q L3)");
    free(X); free(freqs); ref_free(q_pre); ref_free(q_post);
    return d;
}

static int test_sdpa_with_gate(void) {
    printf("[sdpa+output_gate (layer 3 full attention end-to-end)]\n");
    // q,k after rope, v_proj (reshape) -> SDPA -> output_gate -> o_proj
    // Reference: out_oproj_3.bin
    ref_t* q_r = ref_load("src-cpu/refs/q_after_rope_3.bin");      // [L, Nq, Dh]
    ref_t* k_r = ref_load("src-cpu/refs/k_after_rope_3.bin");      // [L, Nkv, Dh]
    ref_t* v_r = ref_load("src-cpu/refs/v_proj_3.bin");            // [L, Nkv*Dh] (flat)
    ref_t* gate_r = ref_load("src-cpu/refs/gate_3.bin");           // [L, Nq*Dh]
    ref_t* ref = ref_load("src-cpu/refs/out_oproj_3.bin");         // [L, hidden]
    size_t L  = (size_t)q_r->shape[0];
    size_t Nq = (size_t)q_r->shape[1];
    size_t Dh = (size_t)q_r->shape[2];
    size_t Nkv = (size_t)k_r->shape[1];
    size_t H  = (size_t)ref->shape[1];

    bf16* Q = calloc(L*Nq*Dh, sizeof(bf16));
    bf16* K = calloc(L*Nkv*Dh, sizeof(bf16));
    bf16* V = calloc(L*Nkv*Dh, sizeof(bf16));
    for (size_t i = 0; i < L*Nq*Dh;  i++) Q[i] = f32_to_bf16(((const float*)q_r->data)[i]);
    for (size_t i = 0; i < L*Nkv*Dh; i++) K[i] = f32_to_bf16(((const float*)k_r->data)[i]);
    for (size_t i = 0; i < L*Nkv*Dh; i++) V[i] = f32_to_bf16(((const float*)v_r->data)[i]);

    size_t Lk = L;
    ref_t* ck = optional_ref("cache_3_0.bin");
    ref_t* cv = optional_ref("cache_3_1.bin");
    if (ck && cv) {
        Lk = (size_t)ck->shape[2];
        free(K); free(V);
        K = calloc(Lk*Nkv*Dh, sizeof(bf16));
        V = calloc(Lk*Nkv*Dh, sizeof(bf16));
        for (size_t t = 0; t < Lk; t++)
            for (size_t h = 0; h < Nkv; h++)
                for (size_t d = 0; d < Dh; d++) {
                    K[(t*Nkv+h)*Dh+d] = f32_to_bf16(((float*)ck->data)[(h*Lk+t)*Dh+d]);
                    V[(t*Nkv+h)*Dh+d] = f32_to_bf16(((float*)cv->data)[(h*Lk+t)*Dh+d]);
                }
    } else if (ck || cv) exit(1);
    ref_free(ck); ref_free(cv);
    bf16* O = calloc(L*Nq*Dh, sizeof(bf16));
    float scale = 1.0f / sqrtf((float)Dh);
    sdpa_gqa_bf16(Q, K, V, O,
                  (uint32_t)L, (uint32_t)Lk,
                  (uint32_t)Nq, (uint32_t)Nkv, (uint32_t)Dh,
                  (uint32_t)(Lk-L), scale, /*causal=*/1);
    ref_t* attn = optional_ref("attn_3.bin");
    int fail = attn ? err_check(O, attn->data, L*Nq*Dh, .05, "same-input SDPA") : 0;
    ref_free(attn);

    // output gate: out = sigmoid(gate) * O (broadcast per element)
    bf16* G = calloc(L*Nq*Dh, sizeof(bf16));
    for (size_t i = 0; i < L*Nq*Dh; i++) G[i] = f32_to_bf16(((const float*)gate_r->data)[i]);
    bf16* Gated = calloc(L*Nq*Dh, sizeof(bf16));
    elemwise_sigmoid_mul_bf16(G, O, Gated, (uint32_t)(L*Nq*Dh));

    // o_proj (q8 linear)
    const uint32_t* W = tens("language_model.model.layers.3.self_attn.o_proj.weight", ST_DTYPE_U32, NULL);
    const bf16* S    = tens("language_model.model.layers.3.self_attn.o_proj.scales",  ST_DTYPE_BF16, NULL);
    const bf16* Bq   = tens("language_model.model.layers.3.self_attn.o_proj.biases",  ST_DTYPE_BF16, NULL);
    bf16* Y = calloc(L*H, sizeof(bf16));
    linear_q8_bf16(Gated, W, S, Bq, NULL, Y,
                   (uint32_t)L, (uint32_t)(Nq*Dh), (uint32_t)H);

    double d = err_check(Y, (const float*)ref->data, L*H, 0.05,
                         "sdpa+gate+o_proj L3");
    free(Q); free(K); free(V); free(O); free(G); free(Gated); free(Y);
    ref_free(q_r); ref_free(k_r); ref_free(v_r); ref_free(gate_r); ref_free(ref);
    return d != 0 || fail;
}

static int test_expert_dense_moe(void) {
    printf("[linear_q8_gather + silu_mul + expert_mix (layer 0 MoE branch)]\n");
    // Validate the per-expert MoE path: gate_proj/up_proj/down_proj + mix.
    ref_t* h_r = ref_load("src-cpu/refs/h_after_post_norm_0.bin");
    ref_t* idx = ref_load("src-cpu/refs/topk_idx_0.bin");
    ref_t* w_r = ref_load("src-cpu/refs/topk_score_0.bin");
    ref_t* gp  = ref_load("src-cpu/refs/gate_proj_0.bin");
    ref_t* up  = ref_load("src-cpu/refs/up_proj_0.bin");
    ref_t* mid = ref_load("src-cpu/refs/swiglu_mid_0.bin");
    ref_t* dn  = ref_load("src-cpu/refs/expert_down_0.bin");
    ref_t* out = ref_load("src-cpu/refs/moe_out_0.bin");

    size_t L = (size_t)h_r->shape[0], H = (size_t)h_r->shape[1];
    size_t Ktop = (size_t)idx->shape[1];
    size_t M = (size_t)gp->shape[2];          // moe_intermediate_size

    bf16* X = calloc(L*H, sizeof(bf16));
    for (size_t i = 0; i < L*H; i++) X[i] = f32_to_bf16(((const float*)h_r->data)[i]);

    // gate_proj: per-expert q8 gather
    const uint32_t* Wg = tens("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", ST_DTYPE_U32, NULL);
    const bf16* Sg = tens("language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales", ST_DTYPE_BF16, NULL);
    const bf16* Bg = tens("language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases", ST_DTYPE_BF16, NULL);
    bf16* GP = calloc(L*Ktop*M, sizeof(bf16));
    linear_q8_gather_bf16(X, Wg, Sg, Bg, (const int32_t*)idx->data, GP,
                          (uint32_t)L, (uint32_t)Ktop, (uint32_t)H, (uint32_t)M);

    // up_proj
    const uint32_t* Wu = tens("language_model.model.layers.0.mlp.switch_mlp.up_proj.weight", ST_DTYPE_U32, NULL);
    const bf16* Su = tens("language_model.model.layers.0.mlp.switch_mlp.up_proj.scales", ST_DTYPE_BF16, NULL);
    const bf16* Bu = tens("language_model.model.layers.0.mlp.switch_mlp.up_proj.biases", ST_DTYPE_BF16, NULL);
    bf16* UP = calloc(L*Ktop*M, sizeof(bf16));
    linear_q8_gather_bf16(X, Wu, Su, Bu, (const int32_t*)idx->data, UP,
                          (uint32_t)L, (uint32_t)Ktop, (uint32_t)H, (uint32_t)M);

    double dgp = err_check(GP, (const float*)gp->data, L*Ktop*M, 0.05, "  gate_proj");
    double dup = err_check(UP, (const float*)up->data, L*Ktop*M, 0.05, "  up_proj");

    // swiglu
    bf16* MID = calloc(L*Ktop*M, sizeof(bf16));
    silu_mul_bf16(GP, UP, MID, (uint32_t)(L*Ktop*M));
    double dmid = err_check(MID, (const float*)mid->data, L*Ktop*M, 0.05, "  swiglu_mid");

    // down_proj
    const uint32_t* Wd = tens("language_model.model.layers.0.mlp.switch_mlp.down_proj.weight", ST_DTYPE_U32, NULL);
    const bf16* Sd = tens("language_model.model.layers.0.mlp.switch_mlp.down_proj.scales", ST_DTYPE_BF16, NULL);
    const bf16* Bd = tens("language_model.model.layers.0.mlp.switch_mlp.down_proj.biases", ST_DTYPE_BF16, NULL);
    bf16* DN = calloc(L*Ktop*H, sizeof(bf16));
    // down_proj is per-token-per-expert: x is [L, K_top, M], weights [E, H, M/4 packed],
    // but our linear_q8_gather expects X[L, K] (single per token). For decoder we need
    // per-(l, kt) row independently. So loop ourselves:
    for (size_t l = 0; l < L; l++) {
        for (size_t kt = 0; kt < Ktop; kt++) {
            // Pretend "L" axis has length 1 and X is the (l, kt) row
            int32_t e = ((const int32_t*)idx->data)[l*Ktop + kt];
            linear_q8_gather_bf16(MID + (l*Ktop + kt)*M,
                                  Wd, Sd, Bd, &e,
                                  DN + (l*Ktop + kt)*H,
                                  1, 1, (uint32_t)M, (uint32_t)H);
        }
    }
    double ddn = err_check(DN, (const float*)dn->data, L*Ktop*H, 0.05, "  expert_down");

    // expert_mix
    bf16* OUT = calloc(L*H, sizeof(bf16));
    expert_mix_bf16(DN, (const float*)w_r->data, OUT, (uint32_t)L, (uint32_t)Ktop, (uint32_t)H);
    double dout = err_check(OUT, (const float*)out->data, L*H, 0.05, "  moe_out");

    int fail = (dgp != 0) | (dup != 0) | (dmid != 0) | (ddn != 0) | (dout != 0);

    free(X); free(GP); free(UP); free(MID); free(DN); free(OUT);
    ref_free(h_r); ref_free(idx); ref_free(w_r);
    ref_free(gp); ref_free(up); ref_free(mid); ref_free(dn); ref_free(out);
    return fail;
}

static int test_conv1d_depthwise(void) {
    printf("[conv1d_depthwise_causal (layer 0 linear-attn)]\n");
    ref_t* qkv = ref_load("src-cpu/refs/qkv_0.bin");
    ref_t* cout = ref_load("src-cpu/refs/conv_out_0.bin");
    size_t L = (size_t)qkv->shape[0], C = (size_t)qkv->shape[1];

    // Conv1d weight: stored as [C, K, 1] in safetensors (after MLX sanitize).
    // Effective per-channel weights [C, K].
    const st_tensor* tw = st_find(g_arch, "language_model.model.layers.0.linear_attn.conv1d.weight");
    if (!tw) { fprintf(stderr, "missing conv1d.weight\n"); exit(1); }
    if (tw->shape[0] != (int64_t)C || tw->shape[1] != 4 || tw->shape[2] != 1) {
        fprintf(stderr, "conv1d.weight unexpected shape %lld,%lld,%lld\n",
                (long long)tw->shape[0], (long long)tw->shape[1], (long long)tw->shape[2]);
        return 1;
    }
    uint32_t K = 4;
    const bf16* W = (const bf16*)tw->data;        // shape [C, 4, 1] -> [C, 4]

    bf16* X = calloc(L*C, sizeof(bf16));
    for (size_t i = 0; i < L*C; i++) X[i] = f32_to_bf16(((const float*)qkv->data)[i]);
    bf16* state = calloc((size_t)(K-1) * C, sizeof(bf16));   // zeros
    ref_t* initial = optional_ref("conv_state_in_0.bin");
    if (initial) {
        if (initial->n_elem != (K-1)*C) exit(1);
        for (size_t i = 0; i < initial->n_elem; i++) state[i] = f32_to_bf16(((float*)initial->data)[i]);
        ref_free(initial);
    }
    bf16* Y = calloc(L*C, sizeof(bf16));
    conv1d_depthwise_causal_bf16(X, state, W, Y, (uint32_t)L, (uint32_t)C, K);
    // Apply SiLU in place (MLX does silu(conv1d(...)) — here we want conv_out post-silu).
    silu_inplace_bf16(Y, (uint32_t)(L*C));

    double d = err_check(Y, (const float*)cout->data, L*C, 0.05,
                         "conv1d + silu (L0)");
    free(X); free(state); free(Y);
    ref_free(qkv); ref_free(cout);
    return d;
}

static int test_gated_delta_step(void) {
    printf("[gated_delta_step (layer 0; Lp tokens)]\n");
    ref_t* q_r = ref_load("src-cpu/refs/q_norm_lin_0.bin");      // [L, Hk, Dk]
    ref_t* k_r = ref_load("src-cpu/refs/k_norm_lin_0.bin");
    ref_t* v_r = ref_load("src-cpu/refs/v_lin_0.bin");           // [L, Hv, Dv]
    ref_t* g_r = ref_load("src-cpu/refs/g_0.bin");               // [L, Hv]
    ref_t* b_r = ref_load("src-cpu/refs/beta_0.bin");
    ref_t* y_r = ref_load("src-cpu/refs/delta_y_0.bin");         // [L, Hv, Dv]
    ref_t* state_r = ref_load("src-cpu/refs/delta_state_0.bin");
    size_t L  = (size_t)q_r->shape[0];
    size_t Hk = (size_t)q_r->shape[1];
    size_t Dk = (size_t)q_r->shape[2];
    size_t Hv = (size_t)v_r->shape[1];
    size_t Dv = (size_t)v_r->shape[2];

    // Run the step sequentially across L tokens.
    bf16 *Q = calloc(L*Hk*Dk, sizeof(bf16));
    bf16 *Kb = calloc(L*Hk*Dk, sizeof(bf16));
    bf16 *V = calloc(L*Hv*Dv, sizeof(bf16));
    float *G = calloc(L*Hv, sizeof(float));
    bf16 *B = calloc(L*Hv, sizeof(bf16));
    for (size_t i = 0; i < L*Hk*Dk; i++) Q[i]  = f32_to_bf16(((const float*)q_r->data)[i]);
    for (size_t i = 0; i < L*Hk*Dk; i++) Kb[i] = f32_to_bf16(((const float*)k_r->data)[i]);
    for (size_t i = 0; i < L*Hv*Dv; i++) V[i]  = f32_to_bf16(((const float*)v_r->data)[i]);
    memcpy(G, g_r->data, L*Hv*sizeof(float));
    for (size_t i = 0; i < L*Hv; i++)    B[i]  = f32_to_bf16(((const float*)b_r->data)[i]);

    float* state = calloc(Hv*Dv*Dk, sizeof(float));    // zero init
    ref_t* initial = optional_ref("delta_state_in_0.bin");
    if (initial) {
        if (initial->n_elem != Hv*Dv*Dk) exit(1);
        memcpy(state, initial->data, initial->n_elem*sizeof(float));
        ref_free(initial);
    }
    bf16*  Y     = calloc(L*Hv*Dv, sizeof(bf16));

    for (size_t l = 0; l < L; l++) {
        gated_delta_step_bf16(Q + l*Hk*Dk, Kb + l*Hk*Dk, V + l*Hv*Dv,
                              G + l*Hv,  B + l*Hv,
                              state,
                              Y + l*Hv*Dv,
                              (uint32_t)Hk, (uint32_t)Hv, (uint32_t)Dk, (uint32_t)Dv);
    }

    double d = err_check(Y, (const float*)y_r->data, L*Hv*Dv, 0.05,
                         "gated_delta_step (sequential L=Lp)");
    int state_fail = state_r->n_elem != Hv*Dv*Dk ||
        check_f32(state, state_r->data, Hv*Dv*Dk, 1e-5, 1e-5,
                  "gated_delta FP32 state");
    free(Q); free(Kb); free(V); free(G); free(B); free(state); free(Y);
    ref_free(q_r); ref_free(k_r); ref_free(v_r); ref_free(g_r); ref_free(b_r); ref_free(y_r);
    ref_free(state_r);
    return d != 0 || state_fail;
}

typedef int (*test_fn)(void);
typedef struct { const char* name; test_fn fn; } test_entry;

static const test_entry TESTS[] = {
    {"embed",       test_embed_gather},
    {"rmsnorm",     test_rmsnorm_input0},
    {"linear_q8",   test_linear_q8_qproj_l3},
    {"silu_mul",    test_silu_mul},
    {"compute_g",   test_compute_g},
    {"recurrent_norms", test_recurrent_norms},
    {"dense",       test_dense_ffn},
    {"softmax_topk", test_softmax_topk},
    {"rope",        test_rope_partial},
    {"sdpa",        test_sdpa_with_gate},
    {"moe",         test_expert_dense_moe},
    {"conv1d",      test_conv1d_depthwise},
    {"gated",       test_gated_delta_step},
};

int main(int argc, char** argv) {
    const char* model_dir = NULL;
    int picked[64] = {0};
    int n_picked = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i+1 < argc) { model_dir = argv[++i]; continue; }
        if (!strcmp(argv[i], "--refs") && i+1 < argc) { ref_directory = argv[++i]; continue; }
        // match by name
        int found = 0;
        for (size_t j = 0; j < sizeof(TESTS)/sizeof(*TESTS); j++) {
            if (!strcmp(argv[i], TESTS[j].name)) {
                if (n_picked >= 64) { fprintf(stderr, "too many test names\n"); return 2; }
                picked[n_picked++] = (int)j; found = 1; break;
            }
        }
        if (!found) { fprintf(stderr, "unknown test: %s\n", argv[i]); return 2; }
    }
    if (n_picked == 0) {
        for (size_t j = 0; j < sizeof(TESTS)/sizeof(*TESTS); j++) picked[n_picked++] = (int)j;
    }

    char* err = NULL;
    if (!model_dir) { fprintf(stderr, "--model is required\n"); return 2; }
    fprintf(stderr, "loading weights from %s ...\n", model_dir);
    g_arch = st_open(model_dir, &err);
    if (!g_arch) { fprintf(stderr, "weights: %s\n", err); return 1; }
    fprintf(stderr, "  loaded %zu tensors\n", st_count(g_arch));

    int fails = 0;
    for (int i = 0; i < n_picked; i++) {
        fails += TESTS[picked[i]].fn();
    }
    st_close(g_arch);
    if (fails) { fprintf(stderr, "%d FAILED\n", fails); return 1; }
    fprintf(stderr, "all tests passed\n");
    return 0;
}
