// Qwen3.8-27B (MLX 8-bit, the same checkpoint the Fortran code reads) inference in C + CUDA.
//
//   ./qwen -m /tmp/certik/model -p "Why is the sky blue?" [-n 512] [--no-think] [-t 0.7]
//
// Pipeline: tokenizer (tokenizer.c) -> chat template -> one token at a time through the model on
// the GPU (the prompt too) -> greedy or temperature sampling -> detokenize and print as we go.
// The whole forward pass for one token (about 600 kernels) is captured once into a CUDA graph.

#include "json.h"
#include "model.h"
#include "safetensors.h"
#include "tokenizer.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    die("CUDA error %s at %s:%d", cudaGetErrorString(e_), __FILE__, __LINE__); } while (0)

static void die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------- config

typedef struct {
    int n_layers, hidden, inter, vocab;
    int nq, nkv, head_dim, rot_dims;      // full attention
    int lin_nk, lin_nv, lin_dk, lin_dv;   // linear attention (Gated DeltaNet)
    int conv_k, full_interval;
    float eps, theta;
} config;

static int get_int(const jval* o, const char* k) {
    const jval* v = json_get(o, k);
    if (!v || v->type != J_NUM) die("config.json: missing %s", k);
    return (int)v->num;
}

static void load_config(const char* dir, config* c) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    static char buf[1 << 20];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    jval* root = json_parse(buf, n);
    if (!root) die("cannot parse %s", path);
    const jval* tc = json_get(root, "text_config");
    if (!tc) tc = root;
    c->n_layers = get_int(tc, "num_hidden_layers");
    c->hidden = get_int(tc, "hidden_size");
    c->inter = get_int(tc, "intermediate_size");
    c->vocab = get_int(tc, "vocab_size");
    c->nq = get_int(tc, "num_attention_heads");
    c->nkv = get_int(tc, "num_key_value_heads");
    c->head_dim = get_int(tc, "head_dim");
    c->lin_nk = get_int(tc, "linear_num_key_heads");
    c->lin_nv = get_int(tc, "linear_num_value_heads");
    c->lin_dk = get_int(tc, "linear_key_head_dim");
    c->lin_dv = get_int(tc, "linear_value_head_dim");
    c->conv_k = get_int(tc, "linear_conv_kernel_dim");
    c->full_interval = get_int(tc, "full_attention_interval");
    c->eps = (float)json_get(tc, "rms_norm_eps")->num;
    const jval* rp = json_get(tc, "rope_parameters");
    c->theta = (float)json_get(rp ? rp : tc, "rope_theta")->num;
    const jval* prf = json_get(rp ? rp : tc, "partial_rotary_factor");
    c->rot_dims = (int)(c->head_dim * (prf ? prf->num : 1.0));
    const jval* q = json_get(root, "quantization");
    if (!q || get_int(q, "bits") != 8 || get_int(q, "group_size") != 64) die("only MLX 8-bit, group 64 is supported");
    json_free(root);
    // The kernels assume these sizes.
    if (c->head_dim != 256 || c->lin_dk != 128 || c->lin_dv != 128 || c->conv_k != 4 || c->hidden > 8192)
        die("unsupported model dimensions");
}

// ---------------------------------------------------------------- weights

typedef struct {
    int is_linear;
    float *in_norm, *post_norm;
    qmat in_proj;   // linear: [qkv | z | b | a]; full: [q+gate | k | v]
    qmat out_proj;  // linear_attn.out_proj or self_attn.o_proj
    qmat gate_up;   // [gate | up]
    qmat down;
    // linear attention
    float *conv_w, *A_log, *dt_bias, *norm_w;
    // full attention
    float *q_norm, *k_norm;
} layer;

typedef struct {
    config c;
    layer* L;
    qmat embed, lm_head;
    float* final_norm;
} model;

static st_model* g_st;
static size_t g_bytes;

static void find(const char* name, st_tensor* t) {
    if (!st_find(g_st, name, t)) die("tensor %s not found", name);
}

static float* upload_vec(const char* name, int n) {
    st_tensor t;
    find(name, &t);
    float* h = malloc(n * sizeof(float));
    if (!strcmp(t.dtype, "BF16") && t.nbytes == (size_t)n * 2) {
        const uint16_t* s = t.data;
        for (int i = 0; i < n; i++) {
            uint32_t u = (uint32_t)s[i] << 16;
            memcpy(&h[i], &u, 4);
        }
    } else if (!strcmp(t.dtype, "F32") && t.nbytes == (size_t)n * 4) memcpy(h, t.data, n * 4);
    else die("tensor %s: unexpected dtype %s / size %zu", name, t.dtype, t.nbytes);
    float* d;
    CK(cudaMalloc((void**)&d, n * sizeof(float)));
    CK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    g_bytes += n * sizeof(float);
    return d;
}

// Uploads q8 matrices with the same K, stacked along N, into one matrix (so one GEMV computes them all).
static qmat upload_q8(const char* const* prefixes, int count) {
    st_tensor w[8], s[8], b[8];
    char name[512];
    int N = 0, K = -1;
    for (int i = 0; i < count; i++) {
        snprintf(name, sizeof(name), "%s.weight", prefixes[i]); find(name, &w[i]);
        snprintf(name, sizeof(name), "%s.scales", prefixes[i]); find(name, &s[i]);
        snprintf(name, sizeof(name), "%s.biases", prefixes[i]); find(name, &b[i]);
        int k = (int)w[i].shape[1] * 4;
        if (strcmp(w[i].dtype, "U32") || w[i].ndim != 2 || strcmp(s[i].dtype, "BF16") || strcmp(b[i].dtype, "BF16") ||
            s[i].shape[0] != w[i].shape[0] || s[i].shape[1] * 64 != k || (K >= 0 && k != K))
            die("%s: not an MLX 8-bit group-64 matrix of the expected shape", prefixes[i]);
        K = k;
        N += (int)w[i].shape[0];
    }
    qmat m = {0};
    m.N = N;
    m.K = K;
    uint8_t *W, *S, *B;
    const size_t sb = (size_t)N * (K / 64) * 2;
    CK(cudaMalloc((void**)&W, (size_t)N * K));
    CK(cudaMalloc((void**)&S, sb));
    CK(cudaMalloc((void**)&B, sb));
    size_t ow = 0, os = 0;
    for (int i = 0; i < count; i++) {
        CK(cudaMemcpy(W + ow, w[i].data, w[i].nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(S + os, s[i].data, s[i].nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(B + os, b[i].data, b[i].nbytes, cudaMemcpyHostToDevice));
        ow += w[i].nbytes;
        os += s[i].nbytes;
    }
    m.W = W;
    m.S = (const uint16_t*)S;
    m.B = (const uint16_t*)B;
    g_bytes += (size_t)N * K + 2 * sb;
    return m;
}

// Uploads [gate; up] with rows interleaved in pairs: g0 g1 u0 u1 g2 g3 u2 u3 ... so that one GEMV block
// (4 rows) produces both inputs of silu(gate_i) * up_i for two outputs i (see k_gemv_swiglu).
static qmat upload_q8_swiglu(const char* gate, const char* up) {
    const char* pr[2] = {gate, up};
    st_tensor t[2][3];
    const char* sfx[3] = {"weight", "scales", "biases"};
    char name[512];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) {
            snprintf(name, sizeof(name), "%s.%s", pr[i], sfx[j]);
            find(name, &t[i][j]);
        }
    const int N = (int)t[0][0].shape[0], K = (int)t[0][0].shape[1] * 4;
    if (strcmp(t[0][0].dtype, "U32") || t[1][0].shape[0] != N || t[1][0].shape[1] * 4 != K || N % 2 ||
        t[0][1].shape[1] * 64 != K)
        die("%s/%s: unexpected shapes", gate, up);
    qmat m = {0};
    m.N = 2 * N;
    m.K = K;
    uint8_t* d[3];
    const size_t row_bytes[3] = {(size_t)K, (size_t)K / 64 * 2, (size_t)K / 64 * 2};
    for (int j = 0; j < 3; j++) {
        CK(cudaMalloc((void**)&d[j], 2 * (size_t)N * row_bytes[j]));
        for (int i = 0; i < 2; i++)   // pairs of rows: source pitch 2 rows, destination pitch 4 rows
            CK(cudaMemcpy2D(d[j] + i * 2 * row_bytes[j], 4 * row_bytes[j], t[i][j].data, 2 * row_bytes[j],
                            2 * row_bytes[j], N / 2, cudaMemcpyHostToDevice));
        g_bytes += 2 * (size_t)N * row_bytes[j];
    }
    m.W = d[0];
    m.S = (const uint16_t*)d[1];
    m.B = (const uint16_t*)d[2];
    return m;
}

static void load_model(const char* dir, model* M) {
    config* c = &M->c;
    load_config(dir, c);
    g_st = st_open(dir);
    if (!g_st) die("cannot open the safetensors checkpoint in %s", dir);
    const char* P = "language_model.model";
    char a[8][256], name[256];
    const char* pa[8];
    for (int i = 0; i < 8; i++) pa[i] = a[i];

    snprintf(a[0], 256, "%s.embed_tokens", P);
    M->embed = upload_q8(pa, 1);
    snprintf(a[0], 256, "language_model.lm_head");
    M->lm_head = upload_q8(pa, 1);
    snprintf(name, sizeof(name), "%s.norm.weight", P);
    M->final_norm = upload_vec(name, c->hidden);

    M->L = calloc(c->n_layers, sizeof(layer));
    for (int l = 0; l < c->n_layers; l++) {
        layer* y = &M->L[l];
        y->is_linear = (l + 1) % c->full_interval != 0;
        fprintf(stderr, "\rloading layer %d/%d", l + 1, c->n_layers);
        snprintf(name, sizeof(name), "%s.layers.%d.input_layernorm.weight", P, l);
        y->in_norm = upload_vec(name, c->hidden);
        snprintf(name, sizeof(name), "%s.layers.%d.post_attention_layernorm.weight", P, l);
        y->post_norm = upload_vec(name, c->hidden);
        if (y->is_linear) {
            const char* n4[] = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"};
            for (int i = 0; i < 4; i++) snprintf(a[i], 256, "%s.layers.%d.linear_attn.%s", P, l, n4[i]);
            y->in_proj = upload_q8(pa, 4);
            snprintf(a[0], 256, "%s.layers.%d.linear_attn.out_proj", P, l);
            y->out_proj = upload_q8(pa, 1);
            const int C = 2 * c->lin_nk * c->lin_dk + c->lin_nv * c->lin_dv;
            snprintf(name, sizeof(name), "%s.layers.%d.linear_attn.conv1d.weight", P, l);
            y->conv_w = upload_vec(name, C * c->conv_k);   // [C, 4, 1]
            snprintf(name, sizeof(name), "%s.layers.%d.linear_attn.A_log", P, l);
            y->A_log = upload_vec(name, c->lin_nv);
            snprintf(name, sizeof(name), "%s.layers.%d.linear_attn.dt_bias", P, l);
            y->dt_bias = upload_vec(name, c->lin_nv);
            snprintf(name, sizeof(name), "%s.layers.%d.linear_attn.norm.weight", P, l);
            y->norm_w = upload_vec(name, c->lin_dv);
        } else {
            const char* n3[] = {"q_proj", "k_proj", "v_proj"};
            for (int i = 0; i < 3; i++) snprintf(a[i], 256, "%s.layers.%d.self_attn.%s", P, l, n3[i]);
            y->in_proj = upload_q8(pa, 3);
            snprintf(a[0], 256, "%s.layers.%d.self_attn.o_proj", P, l);
            y->out_proj = upload_q8(pa, 1);
            snprintf(name, sizeof(name), "%s.layers.%d.self_attn.q_norm.weight", P, l);
            y->q_norm = upload_vec(name, c->head_dim);
            snprintf(name, sizeof(name), "%s.layers.%d.self_attn.k_norm.weight", P, l);
            y->k_norm = upload_vec(name, c->head_dim);
        }
        snprintf(a[0], 256, "%s.layers.%d.mlp.gate_proj", P, l);
        snprintf(a[1], 256, "%s.layers.%d.mlp.up_proj", P, l);
        y->gate_up = upload_q8_swiglu(a[0], a[1]);
        snprintf(a[0], 256, "%s.layers.%d.mlp.down_proj", P, l);
        y->down = upload_q8(pa, 1);
    }
    fprintf(stderr, "\n");
    st_close(g_st);
}

// ---------------------------------------------------------------- state and forward pass

typedef struct {
    int max_ctx;
    int *tok, *pos;                    // device scalars
    resid r;                           // residual stream + fused RMSNorm state
    float *y, *o, *mid, *logits;
    float **conv_state, **ssm_state;   // per linear layer
    attn_params* attn;                 // per full-attention layer (KV cache etc.)
    float* am_val;                     // argmax scratch
    int* am_idx;
    unsigned* counters;                // last-block-done counters, all kept at 0
    volatile int* hist;                // host-mapped: hist[pos] = the token argmax chose at position pos
    int* hist_dev;                     // the same memory, as the device sees it
} state;

static float* dalloc(size_t n) {
    float* p;
    CK(cudaMalloc((void**)&p, n * sizeof(float)));
    CK(cudaMemset(p, 0, n * sizeof(float)));
    g_bytes += n * sizeof(float);
    return p;
}

static void alloc_state(const model* M, state* S, int max_ctx) {
    const config* c = &M->c;
    const int C = 2 * c->lin_nk * c->lin_dk + c->lin_nv * c->lin_dv;
    const int lin_in = C + c->lin_nv * c->lin_dv + 2 * c->lin_nv;
    const int full_in = 2 * c->nq * c->head_dim + 2 * c->nkv * c->head_dim;
    int ymax = c->inter;
    if (lin_in > ymax) ymax = lin_in;
    if (full_in > ymax) ymax = full_in;
    S->max_ctx = max_ctx;
    CK(cudaMalloc((void**)&S->tok, sizeof(int)));
    CK(cudaMalloc((void**)&S->pos, sizeof(int)));
    CK(cudaMalloc((void**)&S->counters, 64 * sizeof(unsigned)));
    CK(cudaMemset(S->counters, 0, 64 * sizeof(unsigned)));
    S->r.H = c->hidden;
    S->r.eps = c->eps;
    S->r.x = dalloc(c->hidden);
    S->r.xw = dalloc(c->hidden);
    CK(cudaMalloc((void**)&S->r.ss, 2 * sizeof(unsigned long long)));
    CK(cudaMemset(S->r.ss, 0, 2 * sizeof(unsigned long long)));
    S->y = dalloc(ymax);
    S->o = dalloc(c->lin_nv * c->lin_dv > c->nq * c->head_dim ? c->lin_nv * c->lin_dv : c->nq * c->head_dim);
    S->mid = dalloc(c->inter);
    S->logits = dalloc(c->vocab);
    CK(cudaHostAlloc((void**)&S->hist, max_ctx * sizeof(int), cudaHostAllocMapped));
    for (int i = 0; i < max_ctx; i++) S->hist[i] = -1;
    CK(cudaHostGetDevicePointer((void**)&S->hist_dev, (void*)S->hist, 0));
    S->am_val = dalloc(256);
    CK(cudaMalloc((void**)&S->am_idx, 256 * sizeof(int)));
    S->conv_state = calloc(c->n_layers, sizeof(float*));
    S->ssm_state = calloc(c->n_layers, sizeof(float*));
    S->attn = calloc(c->n_layers, sizeof(attn_params));
    for (int l = 0; l < c->n_layers; l++) {
        if (M->L[l].is_linear) {
            S->conv_state[l] = dalloc(2 * (size_t)(c->conv_k - 1) * C);
            S->ssm_state[l] = dalloc((size_t)c->lin_nv * c->lin_dk * c->lin_dv);
        } else {
            attn_params* a = &S->attn[l];
            a->qn_w = M->L[l].q_norm;
            a->kn_w = M->L[l].k_norm;
            a->kcache = dalloc((size_t)max_ctx * c->nkv * c->head_dim);
            a->vcache = dalloc((size_t)max_ctx * c->nkv * c->head_dim);
            a->nq = c->nq;
            a->nkv = c->nkv;
            a->rot = c->rot_dims;
            a->theta = c->theta;
            a->eps = c->eps;
        }
    }
}

// Enqueues one token's forward pass: S->tok at position S->pos -> S->logits, argmax into S->tok, pos += 1.
// 5 kernels per layer: RMSNorm, residual adds and SwiGLU are fused into the GEMVs.
static void forward(const model* M, const state* S, cudaStream_t st) {
    const config* c = &M->c;
    resid r = S->r;   // r.k alternates: each producer accumulates ss[k], its consumer reads it
    r.k = 0;
    k_embed(M->embed, S->tok, M->L[0].in_norm, r, st);
    for (int l = 0; l < c->n_layers; l++) {
        const layer* y = &M->L[l];
        const float* next_norm = l + 1 < c->n_layers ? M->L[l + 1].in_norm : M->final_norm;
        k_gemv_norm(y->in_proj, r, S->y, st);                        // y = W_in rmsnorm(x)
        if (y->is_linear)
            k_gdn(S->y, S->conv_state[l], y->conv_w, y->A_log, y->dt_bias, y->norm_w, S->ssm_state[l], S->o,
                  S->pos, c->lin_nv, c->lin_nk, c->eps, st);
        else
            k_attention(S->y, S->attn[l], S->o, S->pos, st);
        r.k = 1;
        k_gemv_resid(y->out_proj, S->o, y->post_norm, r, st);       // x += W_out o
        k_gemv_swiglu(y->gate_up, r, S->mid, st);                    // mid = silu(gate) * up
        r.k = 0;
        k_gemv_resid(y->down, S->mid, next_norm, r, st);             // x += W_down mid
    }
    k_gemv_norm(M->lm_head, r, S->logits, st);
    k_argmax(S->logits, c->vocab, S->tok, S->pos, S->hist_dev, S->am_val, S->am_idx, S->counters + 1, S->r.ss, st);
}

// ---------------------------------------------------------------- sampling

static uint64_t rng_state = 0x2545F4914F6CDD1Dull;
static double rand_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (rng_state >> 11) * (1.0 / 9007199254740992.0);
}

typedef struct { float p; int i; } prob_idx;
static int cmp_prob(const void* a, const void* b) {
    float x = ((const prob_idx*)a)->p, y = ((const prob_idx*)b)->p;
    return (x < y) - (x > y);
}

// Temperature + top-k + top-p sampling on the host.
static int sample(const float* logits, int n, float temp, int top_k, float top_p, prob_idx* buf) {
    int k = top_k > 0 && top_k < n ? top_k : n;
    if (k < n) {   // keep the k largest logits (insertion into a sorted list of k)
        int c = 0;
        for (int i = 0; i < n; i++) {
            if (c == k && logits[i] <= buf[k - 1].p) continue;
            int j = c < k ? c++ : k - 1;
            while (j > 0 && buf[j - 1].p < logits[i]) { buf[j] = buf[j - 1]; j--; }
            buf[j].p = logits[i];
            buf[j].i = i;
        }
    } else {
        for (int i = 0; i < n; i++) { buf[i].p = logits[i]; buf[i].i = i; }
        qsort(buf, n, sizeof(prob_idx), cmp_prob);
    }
    const float m = buf[0].p;
    for (int i = 0; i < k; i++) buf[i].p = expf((buf[i].p - m) / temp);
    double sum = 0;
    for (int i = 0; i < k; i++) sum += buf[i].p;
    double acc = 0;
    int kk = 0;
    while (kk < k) { acc += buf[kk].p / sum; kk++; if (acc >= top_p) break; }
    double r = rand_uniform() * acc * sum, s = 0;
    for (int i = 0; i < kk; i++) { s += buf[i].p; if (s >= r) return buf[i].i; }
    return buf[kk - 1].i;
}

// ---------------------------------------------------------------- main

static void usage(void) {
    fprintf(stderr,
            "usage: qwen [options]\n"
            "  -m DIR        model directory (MLX 8-bit safetensors), default /tmp/certik/model\n"
            "  -p TEXT       prompt (default: \"Hello, who are you?\")\n"
            "  -n N          max new tokens (default 512)\n"
            "  -c N          max context length (default 8192)\n"
            "  -t T          temperature (default 0 = greedy); --top-k K (20), --top-p P (0.95), --seed S\n"
            "  --no-think    disable the thinking block (adds an empty <think></think>)\n"
            "  --raw         use the prompt as is, without the chat template\n"
            "  --ids         print the token ids of the prompt and of the output\n"
            "  --tokenize    only print the prompt's token ids (no GPU, no weights)\n"
            "  --bench N     after the prompt, time N graph replays (GPU time per token), print nothing else\n");
    exit(1);
}

int main(int argc, char** argv) {
    const char* dir = "/tmp/certik/model";
    const char* prompt = "Hello, who are you?";
    int max_new = 512, max_ctx = 8192, think = 1, raw = 0, show_ids = 0, top_k = 20, only_tokenize = 0, bench = 0;
    float temp = 0.f, top_p = 0.95f;
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        int more = i + 1 < argc;
        if (!strcmp(a, "-m") && more) dir = argv[++i];
        else if (!strcmp(a, "-p") && more) prompt = argv[++i];
        else if (!strcmp(a, "-n") && more) max_new = atoi(argv[++i]);
        else if (!strcmp(a, "-c") && more) max_ctx = atoi(argv[++i]);
        else if (!strcmp(a, "-t") && more) temp = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-k") && more) top_k = atoi(argv[++i]);
        else if (!strcmp(a, "--top-p") && more) top_p = (float)atof(argv[++i]);
        else if (!strcmp(a, "--seed") && more) rng_state = strtoull(argv[++i], NULL, 10) * 2654435761ull + 1;
        else if (!strcmp(a, "--no-think")) think = 0;
        else if (!strcmp(a, "--raw")) raw = 1;
        else if (!strcmp(a, "--ids")) show_ids = 1;
        else if (!strcmp(a, "--tokenize")) only_tokenize = 1;
        else if (!strcmp(a, "--bench") && more) bench = atoi(argv[++i]);
        else usage();
    }

    char path[4096];
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    double t0 = now();
    tokenizer* tok = tok_load(path);
    if (!tok) die("cannot load %s", path);

    // Chat template (the generation prompt of the model's chat_template.jinja, one user message).
    char* text = malloc(strlen(prompt) + 256);
    if (raw) strcpy(text, prompt);
    else sprintf(text, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s", prompt,
                 think ? "<think>\n" : "<think>\n\n</think>\n\n");
    int32_t* ids;
    int n_prompt = tok_encode(tok, text, &ids);
    if (show_ids) {
        fprintf(stderr, "prompt ids (%d):", n_prompt);
        for (int i = 0; i < n_prompt; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "\n");
    }
    if (only_tokenize) {
        for (int i = 0; i < n_prompt; i++) printf("%d%s", ids[i], i + 1 < n_prompt ? " " : "\n");
        return 0;
    }
    if (n_prompt + max_new > max_ctx) max_new = max_ctx - n_prompt;
    if (max_new <= 0) die("prompt too long for the context (%d tokens)", n_prompt);
    double t1 = now();

    model M;
    load_model(dir, &M);
    state S;
    alloc_state(&M, &S, max_ctx);
    CK(cudaDeviceSynchronize());
    double t2 = now();
    fprintf(stderr, "tokenizer %.1f s, weights + state %.1f GB in %.1f s, prompt %d tokens\n", t1 - t0,
            g_bytes / 1e9, t2 - t1, n_prompt);

    cudaStream_t st;
    CK(cudaStreamCreate(&st));   // blocking: ordered with the plain cudaMemcpy calls below
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    forward(&M, &S, st);
    CK(cudaStreamEndCapture(st, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));

    int* d_prompt;
    CK(cudaMalloc((void**)&d_prompt, n_prompt * sizeof(int)));
    CK(cudaMemcpy(d_prompt, ids, n_prompt * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemset(S.pos, 0, sizeof(int)));

    // Prefill: the prompt goes through the same single-token graph, one token at a time.
    double t3 = now();
    for (int i = 0; i < n_prompt; i++) {
        CK(cudaMemcpyAsync(S.tok, d_prompt + i, sizeof(int), cudaMemcpyDeviceToDevice, st));
        CK(cudaGraphLaunch(exec, st));
    }
    CK(cudaStreamSynchronize(st));
    double t4 = now();

    if (bench > 0) {   // GPU time per token: back-to-back graph replays
        if (n_prompt + bench > max_ctx) die("--bench: context too small");
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, st));
        for (int i = 0; i < bench; i++) CK(cudaGraphLaunch(exec, st));
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("bench: %d tokens after a %d-token prompt: %.3f ms/token, %.1f tok/s\n", bench, n_prompt, ms / bench,
               1000.0 * bench / ms);
        return 0;
    }

    // Decode. Each graph replay takes the token in S.tok, and its argmax writes the next token to S.tok
    // and to S.hist[position] in host memory. Greedy decoding keeps up to two replays queued and polls
    // S.hist, so the GPU never waits for the host and nothing else runs between the replays. (At the end
    // up to two extra tokens are computed and dropped.) With sampling, the host picks the token from
    // the logits and overwrites S.tok before the next replay.
    float* logits = temp > 0 ? malloc(M.c.vocab * sizeof(float)) : NULL;
    prob_idx* pbuf = temp > 0 ? malloc(M.c.vocab * sizeof(prob_idx)) : NULL;
    int32_t* out_ids = malloc(max_new * sizeof(int32_t));
    int n_out = 0, launched = 0;   // launched: decode replays queued so far
    if (!raw) printf("%s%s", prompt, "\n\n");
    if (!raw && think) printf("<think>\n");
    fflush(stdout);
    double t5 = now();
    for (;;) {
        int next;
        if (temp > 0) {
            CK(cudaMemcpy(logits, S.logits, M.c.vocab * sizeof(float), cudaMemcpyDeviceToHost));
            next = sample(logits, M.c.vocab, temp, top_k, top_p, pbuf);
            CK(cudaMemcpy(S.tok, &next, sizeof(int), cudaMemcpyHostToDevice));
        } else {
            while (launched < n_out + 2 && launched < max_new) {
                CK(cudaGraphLaunch(exec, st));
                launched++;
            }
            const int at = n_prompt - 1 + n_out;   // position whose argmax is the next output token
            while ((next = S.hist[at]) < 0) {}
        }
        out_ids[n_out++] = next;
        if (next == 248046 || next == 248044) break;   // <|im_end|>, <|endoftext|>
        size_t len;
        const char* piece = tok_decode(tok, next, &len);
        fwrite(piece, 1, len, stdout);
        fflush(stdout);
        if (n_out >= max_new) break;
        if (temp > 0) CK(cudaGraphLaunch(exec, st));
    }
    double t6 = now();
    CK(cudaStreamSynchronize(st));
    printf("\n");
    fflush(stdout);
    if (show_ids) {
        fprintf(stderr, "output ids (%d):", n_out);
        for (int i = 0; i < n_out; i++) fprintf(stderr, " %d", out_ids[i]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "prefill: %d tokens in %.3f s (%.1f tok/s); decode: %d tokens in %.3f s (%.1f tok/s)\n",
            n_prompt, t4 - t3, n_prompt / (t4 - t3), n_out - 1, t6 - t5, (n_out - 1) / (t6 - t5));
    return 0;
}
