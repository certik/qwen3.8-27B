// main.c — Qwen 3.5 dense/MoE CPU reference (8-bit MLX-quantized).
//
// Per-layer pipeline mirrors mlx_lm/models/qwen3_5.py + qwen3_next.py.

#include "kernels.h"
#include "safetensors.h"
#include "tokenizer.h"
#include "qwen_chat.h"
#include "utils/json.h"

#include <math.h>
#include <limits.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

static void die(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

static size_t checked_mul(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b) die("allocation size overflow");
    return a * b;
}

static void* alloc_rows(size_t rows, size_t width, size_t element) {
    size_t count = checked_mul(rows, width);
    checked_mul(count, element);
    void* p = calloc(count, element);
    if (count && !p) die("allocation failed");
    return p;
}

static int positive_int(json_val_t obj, const char* key) {
    long n;
    if (json_as_int(json_get(obj, key), &n) || n <= 0 || n > INT_MAX)
        die("config: %s must be a positive integer <= %d", key, INT_MAX);
    return (int)n;
}

static int optional_int(json_val_t obj, const char* key, int fallback) {
    json_val_t v = json_get(obj, key);
    if (json_is_null(v)) return fallback;
    long n;
    if (json_as_int(v, &n) || n < 0 || n > INT_MAX)
        die("config: %s must be a nonnegative integer", key);
    return (int)n;
}

static int optional_bool(json_val_t obj, const char* key, int fallback) {
    json_val_t v = json_get(obj, key);
    if (json_is_null(v)) return fallback;
    int b;
    if (json_as_bool(v, &b)) die("config: %s must be boolean", key);
    return b;
}

static int dimension(size_t value) {
    if (!value || value > INT_MAX) die("config: derived dimension overflow");
    return (int)value;
}

// ---------- config ---------------------------------------------------------

typedef struct {
    int n_layers;
    int hidden;
    int head_dim;
    int n_qheads;
    int n_kvheads;
    int moe_inter;
    int dense_inter;
    int shared_expert_inter;
    int num_experts;
    int top_k;
    int vocab;
    int full_attn_interval;

    int lin_num_k_heads;
    int lin_num_v_heads;
    int lin_key_head_dim;
    int lin_value_head_dim;
    int lin_conv_kernel;
    int lin_key_dim;
    int lin_value_dim;
    int lin_conv_dim;
    int d_rot;

    float partial_rotary_factor;
    float rope_theta;
    float rms_eps;
    int   norm_topk_prob;
    int prompt_capacity;
    int kv_capacity;
    int max_position;
    int eos[16];
    int n_eos;
} cfg_t;

static void load_config(const char* model_dir, cfg_t* c) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    char* err = NULL;
    json_doc_t* doc = json_open_file(path, &err);
    if (!doc) { fprintf(stderr, "config: %s\n", err ? err : "(null)"); exit(1); }
    json_val_t root = json_root(doc);
    json_val_t tc = json_get(root, "text_config");
    if (json_type(tc) != JSON_OBJECT) {
        fprintf(stderr, "config: missing text_config\n"); exit(1);
    }
    if (!json_str_eq(json_get(root, "model_type"), "qwen3_5") &&
        !json_str_eq(json_get(root, "model_type"), "qwen3_5_moe"))
        die("config: unsupported model_type (expected qwen3_5 or qwen3_5_moe)");
    if (!json_str_eq(json_get(tc, "model_type"), "qwen3_5_text") &&
        !json_str_eq(json_get(tc, "model_type"), "qwen3_5_moe_text"))
        die("config: unsupported text model_type");
    json_val_t quant = json_get(root, "quantization");
    if (json_type(quant) != JSON_OBJECT) quant = json_get(root, "quantization_config");
    if (optional_int(quant, "bits", 0) != 8 || optional_int(quant, "group_size", 0) != 64 ||
        (!json_is_null(json_get(quant, "mode")) && !json_str_eq(json_get(quant, "mode"), "affine")))
        die("config: only affine q8 group_size=64 is supported");
    if (optional_bool(tc, "attention_bias", 0) ||
        optional_bool(tc, "tie_word_embeddings", 0) ||
        optional_bool(root, "tie_word_embeddings", 0))
        die("config: attention bias and tied embeddings are unsupported");
    if (!json_is_null(json_get(tc, "hidden_act")) &&
        !json_str_eq(json_get(tc, "hidden_act"), "silu"))
        die("config: only silu FFN activation is supported");
    c->n_layers           = positive_int(tc, "num_hidden_layers");
    c->hidden             = positive_int(tc, "hidden_size");
    c->head_dim           = positive_int(tc, "head_dim");
    c->n_qheads           = positive_int(tc, "num_attention_heads");
    c->n_kvheads          = positive_int(tc, "num_key_value_heads");
    c->num_experts        = optional_int(tc, "num_experts", 0);
    if (c->num_experts) {
        c->moe_inter = positive_int(tc, "moe_intermediate_size");
        c->shared_expert_inter = positive_int(tc, "shared_expert_intermediate_size");
        c->top_k = positive_int(tc, "num_experts_per_tok");
        if (c->top_k > c->num_experts) die("config: top_k exceeds num_experts");
        if (optional_int(tc, "decoder_sparse_step", 1) != 1)
            die("config: only decoder_sparse_step=1 is supported");
    } else {
        c->dense_inter = positive_int(tc, "intermediate_size");
        if (optional_int(tc, "num_experts_per_tok", 0) != 0)
            die("config: dense FFN cannot select experts");
    }
    c->vocab              = positive_int(tc, "vocab_size");
    c->full_attn_interval = positive_int(tc, "full_attention_interval");
    if (c->full_attn_interval < 2 || c->full_attn_interval > c->n_layers)
        die("config: expected a hybrid recurrent/full-attention layer schedule");
    c->lin_num_k_heads    = positive_int(tc, "linear_num_key_heads");
    c->lin_num_v_heads    = positive_int(tc, "linear_num_value_heads");
    c->lin_key_head_dim   = positive_int(tc, "linear_key_head_dim");
    c->lin_value_head_dim = positive_int(tc, "linear_value_head_dim");
    c->lin_conv_kernel    = positive_int(tc, "linear_conv_kernel_dim");
    c->max_position       = positive_int(tc, "max_position_embeddings");
    c->rms_eps            = (float)json_double_or(json_get(tc, "rms_norm_eps"), 1e-6);
    c->norm_topk_prob     = optional_bool(tc, "norm_topk_prob", 1);
    json_val_t rp = json_get(tc, "rope_parameters");
    c->partial_rotary_factor = (float)json_double_or(json_get(rp, "partial_rotary_factor"), 0.25);
    c->rope_theta            = (float)json_double_or(json_get(rp, "rope_theta"), 1e7);

    json_val_t rope_type = json_get(rp, "rope_type");
    if (json_is_null(rope_type)) rope_type = json_get(rp, "type");
    if (!json_is_null(rope_type) && !json_str_eq(rope_type, "default"))
        die("config: only default partial RoPE is supported");
    if (!isfinite(c->rms_eps) || c->rms_eps <= 0 ||
        !isfinite(c->rope_theta) || c->rope_theta <= 0 ||
        !isfinite(c->partial_rotary_factor) || c->partial_rotary_factor <= 0 ||
        c->partial_rotary_factor > 1)
        die("config: invalid normalization or RoPE parameters");
    c->lin_key_dim   = dimension((size_t)c->lin_num_k_heads * c->lin_key_head_dim);
    c->lin_value_dim = dimension((size_t)c->lin_num_v_heads * c->lin_value_head_dim);
    c->lin_conv_dim  = dimension(2 * (size_t)c->lin_key_dim + c->lin_value_dim);
    dimension(2 * (size_t)c->n_qheads * c->head_dim);
    c->d_rot         = (int)((float)c->head_dim * c->partial_rotary_factor);
    if (c->n_qheads % c->n_kvheads || c->lin_num_v_heads % c->lin_num_k_heads ||
        c->lin_conv_kernel < 2 || c->d_rot < 2 || c->d_rot % 2 ||
        c->hidden % 64 || c->lin_value_dim % 64 ||
        (c->n_qheads * c->head_dim) % 64 ||
        (c->num_experts ? c->moe_inter % 64 || c->shared_expert_inter % 64 :
                          c->dense_inter % 64))
        die("config: incompatible head/group/quantization dimensions");
    json_val_t types = json_get(tc, "layer_types");
    if (!json_is_null(types)) {
        if (json_type(types) != JSON_ARRAY || json_arr_len(types) != (size_t)c->n_layers)
            die("config: layer_types must cover every layer");
        for (int li = 0; li < c->n_layers; li++)
            if (!json_str_eq(json_arr_at(types, li),
                    (li+1)%c->full_attn_interval ? "linear_attention" : "full_attention"))
                die("config: layer_types disagrees with full_attention_interval at layer %d", li);
    }
    json_val_t eos = json_get(root, "eos_token_id");
    if (json_is_null(eos)) eos = json_get(tc, "eos_token_id");
    size_t ne = json_type(eos) == JSON_ARRAY ? json_arr_len(eos) : 1;
    if (ne > 16) die("config: too many eos IDs");
    for (size_t i = 0; i < ne; i++) {
        long id;
        if (json_as_int(json_type(eos) == JSON_ARRAY ? json_arr_at(eos, i) : eos, &id) ||
            id < 0 || id >= c->vocab) die("config: invalid eos ID");
        c->eos[c->n_eos++] = (int)id;
    }
    json_close(doc);
}

// ---------- weight accessor ----------------------------------------------

static st_archive* g_arch = NULL;

static const void* W_(const char* fmt, ...) {
    char nm[256];
    va_list ap; va_start(ap, fmt);
    vsnprintf(nm, sizeof(nm), fmt, ap);
    va_end(ap);
    const st_tensor* t = st_find(g_arch, nm);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", nm); exit(1); }
    return t->data;
}

static void expect_tensor(const char* name, st_dtype dtype, int rank, const int64_t* shape) {
    const st_tensor* t = st_find(g_arch, name);
    if (!t) die("missing tensor: %s", name);
    if (t->dtype != dtype || t->rank != rank) die("tensor %s: incompatible dtype/rank", name);
    size_t bytes = st_dtype_size(dtype);
    for (int i = 0; i < rank; i++) {
        if (t->shape[i] != shape[i]) die("tensor %s: dimension %d expected %lld got %lld",
                name, i, (long long)shape[i], (long long)t->shape[i]);
        bytes = checked_mul(bytes, (size_t)shape[i]);
    }
    if (t->nbytes != bytes) die("tensor %s: invalid byte size", name);
}

static void expect_vector(const char* name, int width) {
    int64_t shape[] = {width};
    expect_tensor(name, ST_DTYPE_BF16, 1, shape);
}

static void expect_q8(const char* prefix, int in, int out, int experts) {
    char name[256];
    int64_t shape[] = {experts, out, in/4};
    int rank = experts ? 3 : 2;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    expect_tensor(name, ST_DTYPE_U32, rank, shape + (experts ? 0 : 1));
    shape[2] = in/64;
    snprintf(name, sizeof(name), "%s.scales", prefix);
    expect_tensor(name, ST_DTYPE_BF16, rank, shape + (experts ? 0 : 1));
    snprintf(name, sizeof(name), "%s.biases", prefix);
    expect_tensor(name, ST_DTYPE_BF16, rank, shape + (experts ? 0 : 1));
}

static void validate_weights(const cfg_t* c) {
    char name[256];
    for (size_t i = 0; i < st_count(g_arch); i++)
        if (strstr(st_at(g_arch, i)->name, "mtp."))
            die("weights: MTP/unsanitized normalization shifts are unsupported");
    expect_q8("language_model.model.embed_tokens", c->hidden, c->vocab, 0);
    expect_q8("language_model.lm_head", c->hidden, c->vocab, 0);
    expect_vector("language_model.model.norm.weight", c->hidden);
    for (int li = 0; li < c->n_layers; li++) {
        snprintf(name, sizeof(name), "language_model.model.layers.%d.input_layernorm.weight", li);
        expect_vector(name, c->hidden);
        snprintf(name, sizeof(name), "language_model.model.layers.%d.post_attention_layernorm.weight", li);
        expect_vector(name, c->hidden);
        if ((li+1) % c->full_attn_interval) {
            const char* projection[] = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"};
            int in[] = {c->hidden, c->hidden, c->hidden, c->hidden, c->lin_value_dim};
            int out[] = {c->lin_conv_dim, c->lin_value_dim, c->lin_num_v_heads, c->lin_num_v_heads, c->hidden};
            for (int j = 0; j < 5; j++) {
                snprintf(name, sizeof(name), "language_model.model.layers.%d.linear_attn.%s", li, projection[j]);
                expect_q8(name, in[j], out[j], 0);
            }
            snprintf(name, sizeof(name), "language_model.model.layers.%d.linear_attn.conv1d.weight", li);
            int64_t shape[] = {c->lin_conv_dim, c->lin_conv_kernel, 1};
            expect_tensor(name, ST_DTYPE_BF16, 3, shape);
            snprintf(name, sizeof(name), "language_model.model.layers.%d.linear_attn.A_log", li);
            expect_vector(name, c->lin_num_v_heads);
            snprintf(name, sizeof(name), "language_model.model.layers.%d.linear_attn.dt_bias", li);
            expect_vector(name, c->lin_num_v_heads);
            snprintf(name, sizeof(name), "language_model.model.layers.%d.linear_attn.norm.weight", li);
            expect_vector(name, c->lin_value_head_dim);
        } else {
            const char* projection[] = {"q_proj", "k_proj", "v_proj", "o_proj"};
            int in[] = {c->hidden, c->hidden, c->hidden, c->n_qheads*c->head_dim};
            int out[] = {2*c->n_qheads*c->head_dim, c->n_kvheads*c->head_dim, c->n_kvheads*c->head_dim, c->hidden};
            for (int j = 0; j < 4; j++) {
                snprintf(name, sizeof(name), "language_model.model.layers.%d.self_attn.%s", li, projection[j]);
                expect_q8(name, in[j], out[j], 0);
            }
            snprintf(name, sizeof(name), "language_model.model.layers.%d.self_attn.q_norm.weight", li);
            expect_vector(name, c->head_dim);
            snprintf(name, sizeof(name), "language_model.model.layers.%d.self_attn.k_norm.weight", li);
            expect_vector(name, c->head_dim);
        }
        const char* projection[] = {"gate_proj", "up_proj", "down_proj"};
        for (int j = 0; j < 3; j++) {
            int inter = c->num_experts ? c->moe_inter : c->dense_inter;
            snprintf(name, sizeof(name), "language_model.model.layers.%d.mlp.%s%s", li,
                     c->num_experts ? "switch_mlp." : "", projection[j]);
            expect_q8(name, j == 2 ? inter : c->hidden, j == 2 ? c->hidden : inter, c->num_experts);
            if (c->num_experts) {
                snprintf(name, sizeof(name), "language_model.model.layers.%d.mlp.shared_expert.%s", li, projection[j]);
                expect_q8(name, j == 2 ? c->shared_expert_inter : c->hidden,
                          j == 2 ? c->hidden : c->shared_expert_inter, 0);
            }
        }
        if (c->num_experts) {
            snprintf(name, sizeof(name), "language_model.model.layers.%d.mlp.gate", li);
            expect_q8(name, c->hidden, c->num_experts, 0);
            snprintf(name, sizeof(name), "language_model.model.layers.%d.mlp.shared_expert_gate", li);
            expect_q8(name, c->hidden, 1, 0);
        }
    }
}

static const char* dump_directory;
static char dump_call[4096];

// storage: BF16 widened to F32 (0), native F32 (1), native I32 (2).
static void dump_tensor(const char* op, int li, const void* data, int storage,
                        int rank, const int64_t* shape) {
    if (!dump_directory) return;
    char path[4096];
    int n = li < 0 ? snprintf(path, sizeof(path), "%s/%s.bin", dump_call, op) :
                     snprintf(path, sizeof(path), "%s/%s_%d.bin", dump_call, op, li);
    if (n < 0 || (size_t)n >= sizeof(path)) die("dump path too long");
    FILE* f = fopen(path, "wb");
    if (!f) die("cannot create dump: %s", path);
    uint32_t dtype = storage == 2 ? 1 : 0, ndim = (uint32_t)rank;
    size_t count = 1;
    for (int i = 0; i < rank; i++) count = checked_mul(count, (size_t)shape[i]);
    int fail = fwrite("LLMTNSR1", 1, 8, f) != 8 ||
        fwrite(&dtype, 4, 1, f) != 1 || fwrite(&ndim, 4, 1, f) != 1 ||
        fwrite(shape, 8, (size_t)rank, f) != (size_t)rank;
    if (storage) fail |= fwrite(data, 4, count, f) != count;
    else {
        const bf16* p = data;
        for (size_t i = 0; i < count; i++) {
            float value = bf16_to_f32(p[i]);
            fail |= fwrite(&value, sizeof(value), 1, f) != 1;
        }
    }
    fail |= fclose(f) != 0;
    if (fail) die("dump write failed: %s", path);
}

static void dump_matrix(const char* op, int li, const bf16* data, int rows, int width) {
    int64_t shape[] = {rows, width};
    dump_tensor(op, li, data, 0, 2, shape);
}

static void dump_heads(const char* op, int li, const bf16* data, int rows, int heads, int dim) {
    int64_t shape[] = {rows, heads, dim};
    dump_tensor(op, li, data, 0, 3, shape);
}

// ---------- forward state -------------------------------------------------

static cfg_t g_cfg;
static bf16 *x_buf, *r_buf, *h_buf;
static bf16 *q_full_buf, *k_full_buf, *v_full_buf, *o_full_buf;
static bf16 **k_cache, **v_cache;
static bf16 **conv_state;
static float **delta_state;
static bf16 *qkv_buf, *z_buf, *a_buf, *b_buf, *conv_out_buf;
static bf16 *qlin_buf, *klin_buf, *vlin_buf, *qlin_n_buf, *klin_n_buf;
static float *g_buf;
static bf16 *beta_buf, *delta_y_buf, *normed_buf;
static bf16 *router_logits_buf;
static int32_t *topk_idx_buf;
static float *topk_w_buf;
static bf16 *gp_buf, *up_buf, *mid_buf, *dn_buf, *moe_buf;
static bf16 *sgp_buf, *sup_buf, *sm_buf, *sd_buf, *sg_buf, *sy_buf;
static bf16 *logits_buf;
static float *rope_inv_freqs;
static int cache_position;

// ---------- forward() -----------------------------------------------------

static void forward(int q_off, int Lq) {
    int H = g_cfg.hidden;
    if (q_off != cache_position || Lq <= 0 || Lq > g_cfg.prompt_capacity ||
        (int64_t)q_off + Lq > g_cfg.kv_capacity)
        die("forward: invalid cache position or context capacity");

    for (int li = 0; li < g_cfg.n_layers; li++) {
        int is_linear = ((li + 1) % g_cfg.full_attn_interval) != 0;
        dump_matrix("x_in", li, x_buf, Lq, H);

        rmsnorm_bf16(x_buf,
            (const bf16*)W_("language_model.model.layers.%d.input_layernorm.weight", li),
            h_buf, (uint32_t)Lq, (uint32_t)H, g_cfg.rms_eps);
        dump_matrix("h_after_input_norm", li, h_buf, Lq, H);

        if (is_linear) {
            const char* pref = "language_model.model.layers";
            linear_q8_bf16(h_buf,
                (const uint32_t*)W_("%s.%d.linear_attn.in_proj_qkv.weight", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.in_proj_qkv.scales", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.in_proj_qkv.biases", pref, li),
                NULL, qkv_buf, (uint32_t)Lq, (uint32_t)H, (uint32_t)g_cfg.lin_conv_dim);
            linear_q8_bf16(h_buf,
                (const uint32_t*)W_("%s.%d.linear_attn.in_proj_z.weight", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.in_proj_z.scales", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.in_proj_z.biases", pref, li),
                NULL, z_buf, (uint32_t)Lq, (uint32_t)H, (uint32_t)g_cfg.lin_value_dim);
            linear_q8_bf16(h_buf,
                (const uint32_t*)W_("%s.%d.linear_attn.in_proj_b.weight", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.in_proj_b.scales", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.in_proj_b.biases", pref, li),
                NULL, b_buf, (uint32_t)Lq, (uint32_t)H, (uint32_t)g_cfg.lin_num_v_heads);
            linear_q8_bf16(h_buf,
                (const uint32_t*)W_("%s.%d.linear_attn.in_proj_a.weight", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.in_proj_a.scales", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.in_proj_a.biases", pref, li),
                NULL, a_buf, (uint32_t)Lq, (uint32_t)H, (uint32_t)g_cfg.lin_num_v_heads);
            dump_matrix("qkv", li, qkv_buf, Lq, g_cfg.lin_conv_dim);
            dump_matrix("z", li, z_buf, Lq, g_cfg.lin_value_dim);
            dump_matrix("a", li, a_buf, Lq, g_cfg.lin_num_v_heads);
            dump_matrix("b", li, b_buf, Lq, g_cfg.lin_num_v_heads);
            dump_matrix("conv_state_in", li, conv_state[li], g_cfg.lin_conv_kernel-1, g_cfg.lin_conv_dim);
            int64_t state_shape[] = {g_cfg.lin_num_v_heads, g_cfg.lin_value_head_dim, g_cfg.lin_key_head_dim};
            dump_tensor("delta_state_in", li, delta_state[li], 1, 3, state_shape);

            conv1d_depthwise_causal_bf16(qkv_buf, conv_state[li],
                (const bf16*)W_("%s.%d.linear_attn.conv1d.weight", pref, li),
                conv_out_buf, (uint32_t)Lq, (uint32_t)g_cfg.lin_conv_dim,
                (uint32_t)g_cfg.lin_conv_kernel);
            silu_inplace_bf16(conv_out_buf, (uint32_t)(Lq * g_cfg.lin_conv_dim));
            dump_matrix("conv_out", li, conv_out_buf, Lq, g_cfg.lin_conv_dim);

            for (int l = 0; l < Lq; l++) {
                const bf16* src = conv_out_buf + (size_t)l * g_cfg.lin_conv_dim;
                memcpy(qlin_buf + (size_t)l * g_cfg.lin_key_dim,   src,                              (size_t)g_cfg.lin_key_dim   * sizeof(bf16));
                memcpy(klin_buf + (size_t)l * g_cfg.lin_key_dim,   src + g_cfg.lin_key_dim,          (size_t)g_cfg.lin_key_dim   * sizeof(bf16));
                memcpy(vlin_buf + (size_t)l * g_cfg.lin_value_dim, src + 2 * g_cfg.lin_key_dim,      (size_t)g_cfg.lin_value_dim * sizeof(bf16));
            }
            dump_heads("q_lin", li, qlin_buf, Lq, g_cfg.lin_num_k_heads, g_cfg.lin_key_head_dim);
            dump_heads("k_lin", li, klin_buf, Lq, g_cfg.lin_num_k_heads, g_cfg.lin_key_head_dim);
            dump_heads("v_lin", li, vlin_buf, Lq, g_cfg.lin_num_v_heads, g_cfg.lin_value_head_dim);

            float inv_s = 1.0f / sqrtf((float)g_cfg.lin_key_head_dim);
            rmsnorm_scale_bf16(qlin_buf, qlin_n_buf,
                               (uint32_t)(Lq * g_cfg.lin_num_k_heads),
                               (uint32_t)g_cfg.lin_key_head_dim, 1e-6f, inv_s * inv_s);
            rmsnorm_scale_bf16(klin_buf, klin_n_buf,
                               (uint32_t)(Lq * g_cfg.lin_num_k_heads),
                               (uint32_t)g_cfg.lin_key_head_dim, 1e-6f, inv_s);
            dump_heads("q_norm_lin", li, qlin_n_buf, Lq, g_cfg.lin_num_k_heads, g_cfg.lin_key_head_dim);
            dump_heads("k_norm_lin", li, klin_n_buf, Lq, g_cfg.lin_num_k_heads, g_cfg.lin_key_head_dim);

            memcpy(beta_buf, b_buf, (size_t)Lq * g_cfg.lin_num_v_heads * sizeof(bf16));
            sigmoid_inplace_bf16(beta_buf, (uint32_t)(Lq * g_cfg.lin_num_v_heads));
            compute_g_f32(a_buf,
                (const bf16*)W_("%s.%d.linear_attn.A_log", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.dt_bias", pref, li),
                g_buf, (uint32_t)Lq, (uint32_t)g_cfg.lin_num_v_heads);
            int64_t gate_shape[] = {Lq, g_cfg.lin_num_v_heads};
            dump_tensor("g", li, g_buf, 1, 2, gate_shape);
            dump_matrix("beta", li, beta_buf, Lq, g_cfg.lin_num_v_heads);

            for (int l = 0; l < Lq; l++) {
                gated_delta_step_bf16(
                    qlin_n_buf + (size_t)l * g_cfg.lin_key_dim,
                    klin_n_buf + (size_t)l * g_cfg.lin_key_dim,
                    vlin_buf   + (size_t)l * g_cfg.lin_value_dim,
                    g_buf      + (size_t)l * g_cfg.lin_num_v_heads,
                    beta_buf   + (size_t)l * g_cfg.lin_num_v_heads,
                    delta_state[li],
                    delta_y_buf + (size_t)l * g_cfg.lin_value_dim,
                    (uint32_t)g_cfg.lin_num_k_heads, (uint32_t)g_cfg.lin_num_v_heads,
                    (uint32_t)g_cfg.lin_key_head_dim, (uint32_t)g_cfg.lin_value_head_dim);
            }
            dump_heads("delta_y", li, delta_y_buf, Lq, g_cfg.lin_num_v_heads, g_cfg.lin_value_head_dim);
            dump_tensor("delta_state", li, delta_state[li], 1, 3, state_shape);

            rmsnorm_bf16(delta_y_buf,
                (const bf16*)W_("%s.%d.linear_attn.norm.weight", pref, li),
                normed_buf,
                (uint32_t)(Lq * g_cfg.lin_num_v_heads),
                (uint32_t)g_cfg.lin_value_head_dim, g_cfg.rms_eps);
            silu_mul_f32_gate_bf16(z_buf, normed_buf, normed_buf,
                          (uint32_t)(Lq * g_cfg.lin_value_dim));
            dump_heads("norm_out", li, normed_buf, Lq, g_cfg.lin_num_v_heads, g_cfg.lin_value_head_dim);

            linear_q8_bf16(normed_buf,
                (const uint32_t*)W_("%s.%d.linear_attn.out_proj.weight", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.out_proj.scales", pref, li),
                (const bf16*)W_("%s.%d.linear_attn.out_proj.biases", pref, li),
                NULL, r_buf, (uint32_t)Lq, (uint32_t)g_cfg.lin_value_dim, (uint32_t)H);

        } else {
            const char* pref = "language_model.model.layers";
            int Nq = g_cfg.n_qheads, Nkv = g_cfg.n_kvheads, Dh = g_cfg.head_dim;
            int qproj_n = 2 * Nq * Dh;
            linear_q8_bf16(h_buf,
                (const uint32_t*)W_("%s.%d.self_attn.q_proj.weight", pref, li),
                (const bf16*)W_("%s.%d.self_attn.q_proj.scales", pref, li),
                (const bf16*)W_("%s.%d.self_attn.q_proj.biases", pref, li),
                NULL, q_full_buf, (uint32_t)Lq, (uint32_t)H, (uint32_t)qproj_n);
            linear_q8_bf16(h_buf,
                (const uint32_t*)W_("%s.%d.self_attn.k_proj.weight", pref, li),
                (const bf16*)W_("%s.%d.self_attn.k_proj.scales", pref, li),
                (const bf16*)W_("%s.%d.self_attn.k_proj.biases", pref, li),
                NULL, k_full_buf, (uint32_t)Lq, (uint32_t)H, (uint32_t)(Nkv * Dh));
            linear_q8_bf16(h_buf,
                (const uint32_t*)W_("%s.%d.self_attn.v_proj.weight", pref, li),
                (const bf16*)W_("%s.%d.self_attn.v_proj.scales", pref, li),
                (const bf16*)W_("%s.%d.self_attn.v_proj.biases", pref, li),
                NULL, v_full_buf, (uint32_t)Lq, (uint32_t)H, (uint32_t)(Nkv * Dh));
            dump_matrix("q_proj", li, q_full_buf, Lq, qproj_n);
            dump_matrix("k_proj", li, k_full_buf, Lq, Nkv*Dh);
            dump_matrix("v_proj", li, v_full_buf, Lq, Nkv*Dh);

            for (int l = 0; l < Lq; l++) {
                const bf16* src = q_full_buf + (size_t)l * qproj_n;
                bf16* qdst = qkv_buf + (size_t)l * Nq * Dh;
                bf16* gdst = z_buf   + (size_t)l * Nq * Dh;
                for (int hq = 0; hq < Nq; hq++) {
                    const bf16* row = src + (size_t)hq * 2 * Dh;
                    memcpy(qdst + (size_t)hq * Dh, row,        (size_t)Dh * sizeof(bf16));
                    memcpy(gdst + (size_t)hq * Dh, row + Dh,   (size_t)Dh * sizeof(bf16));
                }
            }
            dump_matrix("gate", li, z_buf, Lq, Nq*Dh);

            rmsnorm_bf16(qkv_buf,
                (const bf16*)W_("%s.%d.self_attn.q_norm.weight", pref, li),
                qkv_buf, (uint32_t)(Lq * Nq), (uint32_t)Dh, g_cfg.rms_eps);
            rmsnorm_bf16(k_full_buf,
                (const bf16*)W_("%s.%d.self_attn.k_norm.weight", pref, li),
                k_full_buf, (uint32_t)(Lq * Nkv), (uint32_t)Dh, g_cfg.rms_eps);
            dump_heads("q_after_qnorm", li, qkv_buf, Lq, Nq, Dh);
            dump_heads("k_after_knorm", li, k_full_buf, Lq, Nkv, Dh);

            rope_partial_inplace_bf16(qkv_buf, rope_inv_freqs,
                (uint32_t)Lq, (uint32_t)Nq, (uint32_t)Dh,
                (uint32_t)g_cfg.d_rot, (uint32_t)q_off);
            rope_partial_inplace_bf16(k_full_buf, rope_inv_freqs,
                (uint32_t)Lq, (uint32_t)Nkv, (uint32_t)Dh,
                (uint32_t)g_cfg.d_rot, (uint32_t)q_off);
            dump_heads("q_after_rope", li, qkv_buf, Lq, Nq, Dh);
            dump_heads("k_after_rope", li, k_full_buf, Lq, Nkv, Dh);

            for (int l = 0; l < Lq; l++) {
                memcpy(k_cache[li] + (size_t)(q_off + l) * Nkv * Dh,
                       k_full_buf  + (size_t)l           * Nkv * Dh,
                       (size_t)Nkv * Dh * sizeof(bf16));
                memcpy(v_cache[li] + (size_t)(q_off + l) * Nkv * Dh,
                       v_full_buf  + (size_t)l           * Nkv * Dh,
                       (size_t)Nkv * Dh * sizeof(bf16));
            }
            int Lk = q_off + Lq;

            sdpa_gqa_bf16(qkv_buf, k_cache[li], v_cache[li], o_full_buf,
                          (uint32_t)Lq, (uint32_t)Lk,
                          (uint32_t)Nq, (uint32_t)Nkv, (uint32_t)Dh,
                          (uint32_t)q_off, 1.0f / sqrtf((float)Dh), /*causal=*/1);
            dump_heads("attn", li, o_full_buf, Lq, Nq, Dh);

            elemwise_sigmoid_mul_bf16(z_buf, o_full_buf, o_full_buf,
                                      (uint32_t)(Lq * Nq * Dh));

            linear_q8_bf16(o_full_buf,
                (const uint32_t*)W_("%s.%d.self_attn.o_proj.weight", pref, li),
                (const bf16*)W_("%s.%d.self_attn.o_proj.scales", pref, li),
                (const bf16*)W_("%s.%d.self_attn.o_proj.biases", pref, li),
                NULL, r_buf, (uint32_t)Lq, (uint32_t)(Nq * Dh), (uint32_t)H);
        }
        dump_matrix("out_oproj", li, r_buf, Lq, H);

        residual_add_bf16(x_buf, r_buf, (uint32_t)(Lq * H));
        dump_matrix("x_after_attn_residual", li, x_buf, Lq, H);

        rmsnorm_bf16(x_buf,
            (const bf16*)W_("language_model.model.layers.%d.post_attention_layernorm.weight", li),
            h_buf, (uint32_t)Lq, (uint32_t)H, g_cfg.rms_eps);
        dump_matrix("h_after_post_norm", li, h_buf, Lq, H);

        if (!g_cfg.num_experts) {
            linear_q8_bf16(h_buf,
                (const uint32_t*)W_("language_model.model.layers.%d.mlp.gate_proj.weight", li),
                (const bf16*)W_("language_model.model.layers.%d.mlp.gate_proj.scales", li),
                (const bf16*)W_("language_model.model.layers.%d.mlp.gate_proj.biases", li),
                NULL, gp_buf, (uint32_t)Lq, (uint32_t)H, (uint32_t)g_cfg.dense_inter);
            linear_q8_bf16(h_buf,
                (const uint32_t*)W_("language_model.model.layers.%d.mlp.up_proj.weight", li),
                (const bf16*)W_("language_model.model.layers.%d.mlp.up_proj.scales", li),
                (const bf16*)W_("language_model.model.layers.%d.mlp.up_proj.biases", li),
                NULL, up_buf, (uint32_t)Lq, (uint32_t)H, (uint32_t)g_cfg.dense_inter);
            silu_mul_bf16(gp_buf, up_buf, mid_buf, (uint32_t)(Lq*g_cfg.dense_inter));
            dump_matrix("gate_proj", li, gp_buf, Lq, g_cfg.dense_inter);
            dump_matrix("up_proj", li, up_buf, Lq, g_cfg.dense_inter);
            dump_matrix("swiglu_mid", li, mid_buf, Lq, g_cfg.dense_inter);
            linear_q8_bf16(mid_buf,
                (const uint32_t*)W_("language_model.model.layers.%d.mlp.down_proj.weight", li),
                (const bf16*)W_("language_model.model.layers.%d.mlp.down_proj.scales", li),
                (const bf16*)W_("language_model.model.layers.%d.mlp.down_proj.biases", li),
                NULL, r_buf, (uint32_t)Lq, (uint32_t)g_cfg.dense_inter, (uint32_t)H);
            dump_matrix("dense_down", li, r_buf, Lq, H);
        } else {
        linear_q8_bf16(h_buf,
            (const uint32_t*)W_("language_model.model.layers.%d.mlp.gate.weight", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.gate.scales", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.gate.biases", li),
            NULL, router_logits_buf,
            (uint32_t)Lq, (uint32_t)H, (uint32_t)g_cfg.num_experts);
        softmax_topk_bf16(router_logits_buf, topk_idx_buf, topk_w_buf,
                          (uint32_t)Lq, (uint32_t)g_cfg.num_experts,
                          (uint32_t)g_cfg.top_k, g_cfg.norm_topk_prob);
        dump_matrix("router_logits", li, router_logits_buf, Lq, g_cfg.num_experts);
        int64_t topk_shape[] = {Lq, g_cfg.top_k};
        dump_tensor("topk_idx", li, topk_idx_buf, 2, 2, topk_shape);
        dump_tensor("topk_score", li, topk_w_buf, 1, 2, topk_shape);

        linear_q8_gather_bf16(h_buf,
            (const uint32_t*)W_("language_model.model.layers.%d.mlp.switch_mlp.gate_proj.weight", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.switch_mlp.gate_proj.scales", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.switch_mlp.gate_proj.biases", li),
            topk_idx_buf, gp_buf,
            (uint32_t)Lq, (uint32_t)g_cfg.top_k, (uint32_t)H, (uint32_t)g_cfg.moe_inter);
        linear_q8_gather_bf16(h_buf,
            (const uint32_t*)W_("language_model.model.layers.%d.mlp.switch_mlp.up_proj.weight", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.switch_mlp.up_proj.scales", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.switch_mlp.up_proj.biases", li),
            topk_idx_buf, up_buf,
            (uint32_t)Lq, (uint32_t)g_cfg.top_k, (uint32_t)H, (uint32_t)g_cfg.moe_inter);
        silu_mul_bf16(gp_buf, up_buf, mid_buf,
                      (uint32_t)(Lq * g_cfg.top_k * g_cfg.moe_inter));
        dump_heads("gate_proj", li, gp_buf, Lq, g_cfg.top_k, g_cfg.moe_inter);
        dump_heads("up_proj", li, up_buf, Lq, g_cfg.top_k, g_cfg.moe_inter);
        dump_heads("swiglu_mid", li, mid_buf, Lq, g_cfg.top_k, g_cfg.moe_inter);

        {
            const uint32_t* Wd = (const uint32_t*)W_("language_model.model.layers.%d.mlp.switch_mlp.down_proj.weight", li);
            const bf16* Sd = (const bf16*)W_("language_model.model.layers.%d.mlp.switch_mlp.down_proj.scales", li);
            const bf16* Bd = (const bf16*)W_("language_model.model.layers.%d.mlp.switch_mlp.down_proj.biases", li);
            for (int l = 0; l < Lq; l++) {
                for (int kt = 0; kt < g_cfg.top_k; kt++) {
                    int32_t e = topk_idx_buf[(size_t)l * g_cfg.top_k + kt];
                    linear_q8_gather_bf16(
                        mid_buf + ((size_t)l * g_cfg.top_k + kt) * g_cfg.moe_inter,
                        Wd, Sd, Bd, &e,
                        dn_buf + ((size_t)l * g_cfg.top_k + kt) * H,
                        1, 1, (uint32_t)g_cfg.moe_inter, (uint32_t)H);
                }
            }
        }
        expert_mix_bf16(dn_buf, topk_w_buf, moe_buf,
                        (uint32_t)Lq, (uint32_t)g_cfg.top_k, (uint32_t)H);
        dump_heads("expert_down", li, dn_buf, Lq, g_cfg.top_k, H);
        dump_matrix("moe_out", li, moe_buf, Lq, H);

        linear_q8_bf16(h_buf,
            (const uint32_t*)W_("language_model.model.layers.%d.mlp.shared_expert.gate_proj.weight", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.shared_expert.gate_proj.scales", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.shared_expert.gate_proj.biases", li),
            NULL, sgp_buf,
            (uint32_t)Lq, (uint32_t)H, (uint32_t)g_cfg.shared_expert_inter);
        linear_q8_bf16(h_buf,
            (const uint32_t*)W_("language_model.model.layers.%d.mlp.shared_expert.up_proj.weight", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.shared_expert.up_proj.scales", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.shared_expert.up_proj.biases", li),
            NULL, sup_buf,
            (uint32_t)Lq, (uint32_t)H, (uint32_t)g_cfg.shared_expert_inter);
        silu_mul_bf16(sgp_buf, sup_buf, sm_buf,
                      (uint32_t)(Lq * g_cfg.shared_expert_inter));
        dump_matrix("shared_gate_proj", li, sgp_buf, Lq, g_cfg.shared_expert_inter);
        dump_matrix("shared_up_proj", li, sup_buf, Lq, g_cfg.shared_expert_inter);
        dump_matrix("shared_swiglu", li, sm_buf, Lq, g_cfg.shared_expert_inter);
        linear_q8_bf16(sm_buf,
            (const uint32_t*)W_("language_model.model.layers.%d.mlp.shared_expert.down_proj.weight", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.shared_expert.down_proj.scales", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.shared_expert.down_proj.biases", li),
            NULL, sd_buf,
            (uint32_t)Lq, (uint32_t)g_cfg.shared_expert_inter, (uint32_t)H);
        linear_q8_bf16(h_buf,
            (const uint32_t*)W_("language_model.model.layers.%d.mlp.shared_expert_gate.weight", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.shared_expert_gate.scales", li),
            (const bf16*)W_("language_model.model.layers.%d.mlp.shared_expert_gate.biases", li),
            NULL, sg_buf, (uint32_t)Lq, (uint32_t)H, 1);
        dump_matrix("shared_down", li, sd_buf, Lq, H);
        dump_matrix("shared_gate", li, sg_buf, Lq, 1);

        sigmoid_inplace_bf16(sg_buf, (uint32_t)Lq);
        for (int l = 0; l < Lq; l++) {
            float g_scalar = bf16_to_f32(sg_buf[l]);
            for (int d = 0; d < H; d++) {
                float v = bf16_to_f32(sd_buf[(size_t)l * H + d]) * g_scalar;
                sy_buf[(size_t)l * H + d] = f32_to_bf16(v);
            }
        }
        dump_matrix("shared_y", li, sy_buf, Lq, H);

        for (size_t i = 0; i < (size_t)Lq * H; i++) {
            float v = bf16_to_f32(moe_buf[i]) + bf16_to_f32(sy_buf[i]);
            r_buf[i] = f32_to_bf16(v);
        }
        }
        dump_matrix("out_mlp", li, r_buf, Lq, H);
        residual_add_bf16(x_buf, r_buf, (uint32_t)(Lq * H));
        dump_matrix("x_after_mlp_residual", li, x_buf, Lq, H);
        if (is_linear) {
            dump_matrix("cache_conv", li, conv_state[li], g_cfg.lin_conv_kernel-1, g_cfg.lin_conv_dim);
            int64_t shape[] = {g_cfg.lin_num_v_heads, g_cfg.lin_value_head_dim, g_cfg.lin_key_head_dim};
            dump_tensor("cache_delta", li, delta_state[li], 1, 3, shape);
        } else {
            dump_heads("cache_k", li, k_cache[li], q_off+Lq, g_cfg.n_kvheads, g_cfg.head_dim);
            dump_heads("cache_v", li, v_cache[li], q_off+Lq, g_cfg.n_kvheads, g_cfg.head_dim);
        }
        if (dump_directory)
            fprintf(stderr, "  context %d layer %d/%d complete\n", q_off+Lq, li+1, g_cfg.n_layers);
    }

    rmsnorm_bf16(x_buf,
                 (const bf16*)W_("language_model.model.norm.weight"),
                 h_buf, (uint32_t)Lq, (uint32_t)H, g_cfg.rms_eps);
    dump_matrix("x_after_final_norm", -1, h_buf, Lq, H);
    linear_q8_bf16(h_buf,
        (const uint32_t*)W_("language_model.lm_head.weight"),
        (const bf16*)W_("language_model.lm_head.scales"),
        (const bf16*)W_("language_model.lm_head.biases"),
        NULL, logits_buf, (uint32_t)Lq, (uint32_t)H, (uint32_t)g_cfg.vocab);
    dump_matrix("logits", -1, logits_buf, Lq, g_cfg.vocab);
    cache_position += Lq;
}

static void alloc_buffers(void) {
    int H = g_cfg.hidden, M = g_cfg.prompt_capacity;
    x_buf   = alloc_rows(M, H, sizeof(bf16));
    r_buf   = alloc_rows(M, H, sizeof(bf16));
    h_buf   = alloc_rows(M, H, sizeof(bf16));

    int Nq = g_cfg.n_qheads, Nkv = g_cfg.n_kvheads, Dh = g_cfg.head_dim;
    int qwidth = Nq*Dh;
    q_full_buf = alloc_rows(M, 2*qwidth, sizeof(bf16));
    k_full_buf = alloc_rows(M, (size_t)Nkv*Dh, sizeof(bf16));
    v_full_buf = alloc_rows(M, (size_t)Nkv*Dh, sizeof(bf16));
    o_full_buf = alloc_rows(M, qwidth, sizeof(bf16));

    qkv_buf      = alloc_rows(M, g_cfg.lin_conv_dim > qwidth ? g_cfg.lin_conv_dim : qwidth, sizeof(bf16));
    z_buf        = alloc_rows(M, g_cfg.lin_value_dim > qwidth ? g_cfg.lin_value_dim : qwidth, sizeof(bf16));
    a_buf        = alloc_rows(M, g_cfg.lin_num_v_heads, sizeof(bf16));
    b_buf        = alloc_rows(M, g_cfg.lin_num_v_heads, sizeof(bf16));
    conv_out_buf = alloc_rows(M, g_cfg.lin_conv_dim, sizeof(bf16));
    qlin_buf     = alloc_rows(M, g_cfg.lin_key_dim, sizeof(bf16));
    klin_buf     = alloc_rows(M, g_cfg.lin_key_dim, sizeof(bf16));
    vlin_buf     = alloc_rows(M, g_cfg.lin_value_dim, sizeof(bf16));
    qlin_n_buf   = alloc_rows(M, g_cfg.lin_key_dim, sizeof(bf16));
    klin_n_buf   = alloc_rows(M, g_cfg.lin_key_dim, sizeof(bf16));
    g_buf        = alloc_rows(M, g_cfg.lin_num_v_heads, sizeof(float));
    beta_buf     = alloc_rows(M, g_cfg.lin_num_v_heads, sizeof(bf16));
    delta_y_buf  = alloc_rows(M, g_cfg.lin_value_dim, sizeof(bf16));
    normed_buf   = alloc_rows(M, g_cfg.lin_value_dim, sizeof(bf16));

    size_t ffn_width = g_cfg.num_experts ? checked_mul(g_cfg.top_k, g_cfg.moe_inter) : (size_t)g_cfg.dense_inter;
    gp_buf = alloc_rows(M, ffn_width, sizeof(bf16));
    up_buf = alloc_rows(M, ffn_width, sizeof(bf16));
    mid_buf = alloc_rows(M, ffn_width, sizeof(bf16));
    if (g_cfg.num_experts) {
        router_logits_buf = alloc_rows(M, g_cfg.num_experts, sizeof(bf16));
        topk_idx_buf = alloc_rows(M, g_cfg.top_k, sizeof(int32_t));
        topk_w_buf = alloc_rows(M, g_cfg.top_k, sizeof(float));
        dn_buf = alloc_rows(M, checked_mul(g_cfg.top_k, H), sizeof(bf16));
        moe_buf = alloc_rows(M, H, sizeof(bf16));
        sgp_buf = alloc_rows(M, g_cfg.shared_expert_inter, sizeof(bf16));
        sup_buf = alloc_rows(M, g_cfg.shared_expert_inter, sizeof(bf16));
        sm_buf = alloc_rows(M, g_cfg.shared_expert_inter, sizeof(bf16));
        sd_buf = alloc_rows(M, H, sizeof(bf16));
        sg_buf = alloc_rows(M, 1, sizeof(bf16));
        sy_buf = alloc_rows(M, H, sizeof(bf16));
    }

    logits_buf = alloc_rows(M, g_cfg.vocab, sizeof(bf16));

    k_cache = alloc_rows(g_cfg.n_layers, 1, sizeof(bf16*));
    v_cache = alloc_rows(g_cfg.n_layers, 1, sizeof(bf16*));
    conv_state  = alloc_rows(g_cfg.n_layers, 1, sizeof(bf16*));
    delta_state = alloc_rows(g_cfg.n_layers, 1, sizeof(float*));
    for (int li = 0; li < g_cfg.n_layers; li++) {
        int is_linear = ((li + 1) % g_cfg.full_attn_interval) != 0;
        if (is_linear) {
            conv_state[li]  = alloc_rows(g_cfg.lin_conv_kernel-1, g_cfg.lin_conv_dim, sizeof(bf16));
            delta_state[li] = alloc_rows(g_cfg.lin_num_v_heads,
                checked_mul(g_cfg.lin_value_head_dim, g_cfg.lin_key_head_dim), sizeof(float));
        } else {
            k_cache[li] = alloc_rows(g_cfg.kv_capacity, checked_mul(Nkv, Dh), sizeof(bf16));
            v_cache[li] = alloc_rows(g_cfg.kv_capacity, checked_mul(Nkv, Dh), sizeof(bf16));
        }
    }

    int Dh_rope = g_cfg.d_rot / 2;
    rope_inv_freqs = alloc_rows(Dh_rope, 1, sizeof(float));
    for (int d = 0; d < Dh_rope; d++) {
        rope_inv_freqs[d] = 1.0f / powf(g_cfg.rope_theta, (float)(2*d) / (float)g_cfg.d_rot);
    }
}

static int32_t* load_ids(const char* path, int vocab, int* length) {
    FILE* f = fopen(path, "rb");
    if (!f) die("input IDs: cannot open %s", path);
    int ch;
    do { ch = fgetc(f); } while (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t');
    if (ch != '[') die("input IDs: expected a JSON integer array in %s", path);
    size_t cap = 16, n = 0;
    int32_t* ids = alloc_rows(cap, 1, sizeof(*ids));
    for (;;) {
        do { ch = fgetc(f); } while (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t');
        if (ch < '0' || ch > '9') die("input IDs: empty array or invalid integer in %s", path);
        int first = ch, digits = 0;
        uint64_t value = 0;
        do {
            if (digits++ && first == '0') die("input IDs: leading zero in %s", path);
            value = value * 10 + (unsigned)(ch-'0');
            if (value >= (uint64_t)vocab) die("input IDs: out-of-vocabulary integer in %s", path);
            ch = fgetc(f);
        } while (ch >= '0' && ch <= '9');
        if (n == INT_MAX) die("input IDs: too many tokens");
        if (n == cap) {
            cap = checked_mul(cap, 2);
            int32_t* p = realloc(ids, checked_mul(cap, sizeof(*ids)));
            if (!p) die("input IDs: allocation failed");
            ids = p;
        }
        ids[n++] = (int32_t)value;
        while (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t') ch = fgetc(f);
        if (ch == ']') break;
        if (ch != ',') die("input IDs: expected comma or closing bracket in %s", path);
    }
    do { ch = fgetc(f); } while (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t');
    if (ch != EOF || ferror(f)) die("input IDs: trailing data/read error in %s", path);
    fclose(f);
    *length = (int)n;
    return ids;
}

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec*1e-9;
}

static uint64_t hash_bytes(uint64_t hash, const void* data, size_t n) {
    const unsigned char* p = data;
    for (size_t i = 0; i < n; i++) hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

static uint64_t cache_hash(void) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (int li = 0; li < g_cfg.n_layers; li++) {
        if (conv_state[li]) {
            h = hash_bytes(h, conv_state[li], checked_mul((size_t)(g_cfg.lin_conv_kernel-1)*g_cfg.lin_conv_dim, sizeof(bf16)));
            h = hash_bytes(h, delta_state[li], checked_mul((size_t)g_cfg.lin_num_v_heads*g_cfg.lin_value_head_dim*g_cfg.lin_key_head_dim, sizeof(float)));
        } else {
            size_t bytes = checked_mul((size_t)cache_position*g_cfg.n_kvheads*g_cfg.head_dim, sizeof(bf16));
            h = hash_bytes(h, k_cache[li], bytes);
            h = hash_bytes(h, v_cache[li], bytes);
        }
    }
    return h;
}

static void reset_cache(void) {
    for (int li = 0; li < g_cfg.n_layers; li++) {
        if (conv_state[li]) {
            memset(conv_state[li], 0, (size_t)(g_cfg.lin_conv_kernel-1)*g_cfg.lin_conv_dim*sizeof(bf16));
            memset(delta_state[li], 0, (size_t)g_cfg.lin_num_v_heads*g_cfg.lin_value_head_dim*g_cfg.lin_key_head_dim*sizeof(float));
        } else {
            size_t bytes = (size_t)g_cfg.kv_capacity*g_cfg.n_kvheads*g_cfg.head_dim*sizeof(bf16);
            memset(k_cache[li], 0, bytes);
            memset(v_cache[li], 0, bytes);
        }
    }
    cache_position = 0;
}

static void begin_dump(int step) {
    if (!dump_directory) return;
    int n = step == 0 ? snprintf(dump_call, sizeof(dump_call), "%s", dump_directory) :
        (step < 0 ? snprintf(dump_call, sizeof(dump_call), "%s/reset", dump_directory) :
                    snprintf(dump_call, sizeof(dump_call), "%s/decode-%03d", dump_directory, step));
    if (n < 0 || (size_t)n >= sizeof(dump_call)) die("dump path too long");
    if (mkdir(dump_call, 0755)) die("cannot create new dump directory %s: %s", dump_call, strerror(errno));
}

static void embed(const int32_t* ids, int count) {
    embed_dequant_gather_q8(ids,
        (const uint32_t*)W_("language_model.model.embed_tokens.weight"),
        (const bf16*)W_("language_model.model.embed_tokens.scales"),
        (const bf16*)W_("language_model.model.embed_tokens.biases"),
        x_buf, (uint32_t)count, (uint32_t)g_cfg.hidden);
    dump_matrix("x_after_embed", -1, x_buf, count, g_cfg.hidden);
}

static int32_t last_argmax(int rows) {
    for (size_t i = 0; i < (size_t)rows*g_cfg.vocab; i++)
        if (!isfinite(bf16_to_f32(logits_buf[i]))) die("forward: non-finite logits");
    int32_t id;
    argmax_bf16(logits_buf + (size_t)(rows-1)*g_cfg.vocab, &id, (uint32_t)g_cfg.vocab);
    return id;
}

int main(int argc, char** argv) {
    double start = now_seconds();
    const char *prompt = NULL, *sysmsg = NULL, *input_file = NULL, *teacher_file = NULL;
    const char* model_dir = "/Users/ondrej/.cache/lm-studio/models/unsloth/Qwen3.6-35B-A3B-MLX-8bit";
    const char* result_path = NULL;
    int max_tokens = 8, fixed_steps = 0, reset_check = 0, check_config = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--system") && i+1 < argc) sysmsg = argv[++i];
        else if (!strcmp(argv[i], "--model") && i+1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--input-ids") && i+1 < argc) input_file = argv[++i];
        else if (!strcmp(argv[i], "--teacher-ids") && i+1 < argc) teacher_file = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i+1 < argc) dump_directory = argv[++i];
        else if (!strcmp(argv[i], "--result") && i+1 < argc) result_path = argv[++i];
        else if (!strcmp(argv[i], "--fixed-steps")) fixed_steps = 1;
        else if (!strcmp(argv[i], "--reset-check")) reset_check = 1;
        else if (!strcmp(argv[i], "--check-config")) check_config = 1;
        else if (!strcmp(argv[i], "--max-tokens") && i+1 < argc) {
            char* end;
            errno = 0;
            long n = strtol(argv[++i], &end, 10);
            if (errno || *end || n <= 0 || n > INT_MAX) die("--max-tokens must be a positive integer");
            max_tokens = (int)n;
        } else die("unknown or incomplete argument: %s", argv[i]);
    }
    load_config(model_dir, &g_cfg);
    fprintf(stderr, "config: %d layers hidden=%d q_width=%d recurrent_value=%d %s inter=%d\n",
            g_cfg.n_layers, g_cfg.hidden, g_cfg.n_qheads*g_cfg.head_dim, g_cfg.lin_value_dim,
            g_cfg.num_experts ? "MoE" : "dense", g_cfg.num_experts ? g_cfg.moe_inter : g_cfg.dense_inter);
    if (check_config) return 0;
    if ((!prompt && !input_file) || (prompt && input_file) || (sysmsg && input_file))
        die("provide exactly one of --prompt TEXT or --input-ids JSON_ARRAY (no --system with raw IDs)");
    if (teacher_file && !fixed_steps) die("--teacher-ids requires --fixed-steps");
    char* err = NULL;
    tokenizer_t* tk = NULL;
    int32_t* prompt_ids;
    int Lp, n_teacher = 0;
    if (input_file) prompt_ids = load_ids(input_file, g_cfg.vocab, &Lp);
    else {
        if (!g_cfg.num_experts)
            die("dense text/template/NFC input is not implemented; use frozen --input-ids");
        tk = tk_load("src-cpu/tokenizer.bin", &err);
        if (!tk) die("tokenizer load: %s", err ? err : "unknown error");
        // The legacy pure-C tokenizer/chat builder is retained for old MoE text mode.
        prompt_ids = alloc_rows(g_cfg.max_position, 1, sizeof(*prompt_ids));
        Lp = qwen_chat_build_prompt(tk, sysmsg, prompt, prompt_ids, g_cfg.max_position);
        if (Lp <= 0) die("invalid or oversized legacy prompt");
        for (int i = 0; i < Lp; i++)
            if (prompt_ids[i] < 0 || prompt_ids[i] >= g_cfg.vocab) die("legacy tokenizer ID out of vocabulary");
    }
    int32_t* teacher_ids = teacher_file ? load_ids(teacher_file, g_cfg.vocab, &n_teacher) : NULL;
    if (teacher_ids && n_teacher < max_tokens-1) die("insufficient teacher continuation IDs");
    int64_t capacity = (int64_t)Lp + max_tokens - 1;
    if (capacity > g_cfg.max_position || capacity > INT_MAX) die("requested context exceeds model capacity");
    g_cfg.prompt_capacity = Lp;
    g_cfg.kv_capacity = (int)capacity;
    size_t widths[] = {g_cfg.hidden, 2*(size_t)g_cfg.n_qheads*g_cfg.head_dim,
        g_cfg.lin_conv_dim, g_cfg.lin_value_dim, g_cfg.vocab,
        g_cfg.num_experts ? checked_mul(g_cfg.top_k, g_cfg.moe_inter) : (size_t)g_cfg.dense_inter,
        g_cfg.num_experts ? (size_t)g_cfg.shared_expert_inter : 1};
    for (size_t i = 0; i < sizeof(widths)/sizeof(*widths); i++)
        if ((size_t)Lp > INT_MAX / widths[i]) die("workspace exceeds scalar kernel index capacity");
    fprintf(stderr, "prompt=%d activation_capacity=%d KV_capacity=%d; loading weights...\n",
            Lp, g_cfg.prompt_capacity, g_cfg.kv_capacity);
    g_arch = st_open(model_dir, &err);
    if (!g_arch) die("weights: %s", err ? err : "unknown error");
    validate_weights(&g_cfg);
    alloc_buffers();
    double load_seconds = now_seconds()-start;
    int32_t* generated = alloc_rows(max_tokens, 1, sizeof(*generated));
    bf16* prefill_logits = reset_check ? alloc_rows(Lp, g_cfg.vocab, sizeof(bf16)) : NULL;
    double prefill_start = now_seconds();
    begin_dump(0);
    embed(prompt_ids, Lp);
    forward(0, Lp);
    int32_t next_id = last_argmax(Lp);
    double prefill_seconds = now_seconds()-prefill_start;
    uint64_t prefill_hash = cache_hash();
    if (reset_check) memcpy(prefill_logits, logits_buf, (size_t)Lp*g_cfg.vocab*sizeof(bf16));
    fprintf(stderr, "prefill/TTFT: %.6fs, effective %.6f prompt tok/s; first ID=%d\n",
            prefill_seconds, Lp/prefill_seconds, next_id);
    int n_out = 0;
    double decode_start = now_seconds();
    for (int step = 0; step < max_tokens; step++) {
        generated[n_out++] = next_id;
        fprintf(stderr, "generated[%d]=%d\n", step, next_id);
        if (tk) {
            int len;
            const char* bytes = tk_token_bytes(tk, next_id, &len);
            if (bytes && len > 0) { fwrite(bytes, 1, (size_t)len, stdout); fflush(stdout); }
        }
        int eos = 0;
        for (int i = 0; i < g_cfg.n_eos; i++) eos |= next_id == g_cfg.eos[i];
        if (step+1 == max_tokens || (eos && !fixed_steps)) break;
        int32_t input = teacher_ids ? teacher_ids[step] : next_id;
        begin_dump(step+1);
        embed(&input, 1);
        forward(Lp+step, 1);
        next_id = last_argmax(1);
    }
    double decode_seconds = now_seconds()-decode_start;
    double inference_seconds = now_seconds()-prefill_start;
    int processed = cache_position;
    double reset_seconds = 0;
    if (reset_check) {
        double reset_start = now_seconds();
        reset_cache();
        begin_dump(-1);
        embed(prompt_ids, Lp);
        forward(0, Lp);
        last_argmax(Lp);
        if (cache_hash() != prefill_hash ||
            memcmp(prefill_logits, logits_buf, (size_t)Lp*g_cfg.vocab*sizeof(bf16)))
            die("reset/reuse: cache or logits differs from original prefill");
        reset_seconds = now_seconds()-reset_start;
        fprintf(stderr, "reset/reuse: all logits and all caches byte-exact\n");
    }
    double wall_seconds = now_seconds()-start;
    fprintf(stderr, "generated_ids: [");
    for (int i = 0; i < n_out; i++) fprintf(stderr, "%s%d", i ? ", " : "", generated[i]);
    fprintf(stderr, "]\n");
    if (result_path) {
        FILE* f = fopen(result_path, "wx");
        if (!f) die("cannot create new result file %s", result_path);
        fprintf(f, "{\"schema\":2,\"input_ids\":[");
        for (int i = 0; i < Lp; i++) fprintf(f, "%s%d", i ? "," : "", prompt_ids[i]);
        fprintf(f, "],\"generated_ids\":[");
        for (int i = 0; i < n_out; i++) fprintf(f, "%s%d", i ? "," : "", generated[i]);
        fprintf(f, "],\"consumed_decode_ids\":[");
        for (int i = 0; i < n_out-1; i++)
            fprintf(f, "%s%d", i ? "," : "", teacher_ids ? teacher_ids[i] : generated[i]);
        fprintf(f, "],\"arithmetic_profile\":\"mlx-0.31.2-m4max\","
                   "\"rms_accumulation\":\"compensated-fp32\","
                   "\"requested_generated\":%d,\"actual_generated\":%d,"
                   "\"fixed_steps\":%s,\"teacher_forced\":%s,\"activation_capacity\":%d,"
                   "\"KV_capacity\":%d,\"inference_cache_tokens_processed\":%d,"
                   "\"cache_tokens_processed\":%d,\"inference_forward_calls\":%d,"
                   "\"reset_forward_calls\":%d,\"model_forward_calls\":%d,"
                   "\"extra_unemitted_decode_forwards\":0,\"reset_checked\":%s,"
                   "\"reset_cache_and_logits_exact\":%s,\"reset_seconds\":%.9f,"
                   "\"startup_seconds\":%.9f,\"prefill_ttft_seconds\":%.9f,"
                   "\"effective_prefill_ttft_tps\":%.9f,\"decode_seconds\":%.9f,"
                   "\"decode_tps\":%.9f,\"inference_seconds\":%.9f,\"process_wall_seconds\":%.9f,"
                   "\"dump_io_included\":%s,\"throughput_valid\":false}\n",
                max_tokens, n_out, fixed_steps ? "true" : "false", teacher_ids ? "true" : "false",
                Lp, g_cfg.kv_capacity, processed, processed+(reset_check ? Lp : 0),
                n_out, reset_check ? 1 : 0, n_out+(reset_check ? 1 : 0), reset_check ? "true" : "false",
                reset_check ? "true" : "false", reset_seconds, load_seconds, prefill_seconds,
                Lp/prefill_seconds, decode_seconds, (n_out-1)/decode_seconds,
                inference_seconds, wall_seconds, dump_directory ? "true" : "false");
        if (ferror(f) || fclose(f)) die("result write failed");
    }
    fprintf(stderr, "startup %.6fs; inference %.6fs; decode %.6f tok/s; wall %.6fs (reset %.6fs)\n",
            load_seconds, inference_seconds, (n_out-1)/decode_seconds, wall_seconds, reset_seconds);
    free(prefill_logits); free(generated); free(teacher_ids); free(prompt_ids);
    st_close(g_arch);
    tk_free(tk);
    return 0;
}
