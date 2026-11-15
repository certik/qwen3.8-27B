#include "../kernels.h"
#include "../safetensors.h"
#include "load_ref.h"
#include "compare.h"

static ref_t* load_at(const char* directory, const char* name) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", directory, name);
    return ref_load(path);
}

static bf16* narrow(const ref_t* r) {
    bf16* p = calloc(r->n_elem, sizeof(*p));
    if (!p || r->dtype) exit(1);
    for (size_t i = 0; i < r->n_elem; i++) p[i] = f32_to_bf16(((float*)r->data)[i]);
    return p;
}

int main(int argc, char** argv) {
    const char *model = NULL, *inputs = NULL, *refs = NULL;
    int rms_layer = -1;
    int rms_all = 0;
    int q8_gates = 0;
    int q8_partials = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i+1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--inputs") && i+1 < argc) inputs = argv[++i];
        else if (!strcmp(argv[i], "--refs") && i+1 < argc) refs = argv[++i];
        else if (!strcmp(argv[i], "--rms-layer") && i+1 < argc) rms_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rms-all")) rms_all = 1;
        else if (!strcmp(argv[i], "--q8-gates")) q8_gates = 1;
        else if (!strcmp(argv[i], "--q8-partials")) q8_partials = 1;
        else return 2;
    }
    if (!model || !inputs || !refs) return 2;
    char* error = NULL;
    st_archive* archive = st_open(model, &error);
    if (!archive) { fprintf(stderr, "%s\n", error); return 1; }
    if (q8_partials) {
        int fail = 0;
        for (int layer = 0; layer < 64; layer++) {
            int linear = (layer+1)%4 != 0;
            for (int op = 0; op < (linear ? 1 : 3); op++) {
                char name[256], prefix[256];
                const char* projection = op == 0 ? (linear ? "out_proj" : "o_proj") :
                                                (op == 1 ? "k_proj" : "v_proj");
                snprintf(name, sizeof(name), "%s_%d.bin",
                         op ? "h_after_input_norm" : (linear ? "norm_out" : "attn"), layer);
                ref_t* input = load_at(inputs, name);
                bf16* x = narrow(input);
                if (!linear && !op) {
                    snprintf(name, sizeof(name), "gate_%d.bin", layer);
                    ref_t* gate = load_at(inputs, name);
                    bf16* g = narrow(gate);
                    elemwise_sigmoid_mul_bf16(g, x, x, (uint32_t)input->n_elem);
                    free(g); ref_free(gate);
                }
                snprintf(name, sizeof(name), "%s_%d.bin", op ? projection : "out_oproj", layer);
                ref_t* expected = load_at(refs, name);
                bf16* y = calloc(expected->n_elem, sizeof(*y));
                snprintf(prefix, sizeof(prefix), "language_model.model.layers.%d.%s.%s", layer,
                         linear ? "linear_attn" : "self_attn", projection);
                snprintf(name, sizeof(name), "%s.weight", prefix);
                const st_tensor* w = st_find(archive, name);
                snprintf(name, sizeof(name), "%s.scales", prefix);
                const st_tensor* s = st_find(archive, name);
                snprintf(name, sizeof(name), "%s.biases", prefix);
                const st_tensor* b = st_find(archive, name);
                if (!w || !s || !b || !y) return 1;
                uint32_t rows = (uint32_t)input->shape[0];
                linear_q8_bf16(x, w->data, s->data, b->data, NULL, y, rows,
                               (uint32_t)(input->n_elem/rows), (uint32_t)expected->shape[1]);
                snprintf(name, sizeof(name), "L%d %s", layer, projection);
                fail |= err_check(y, expected->data, expected->n_elem,
                                  !linear && !op ? .05 : .03, name);
                free(x); free(y); ref_free(input); ref_free(expected);
            }
        }
        st_close(archive);
        return fail;
    }
    if (q8_gates) {
        int fail = 0;
        for (int layer = 0; layer < 64; layer++) {
            if ((layer+1)%4 == 0) continue;
            char name[256];
            snprintf(name, sizeof(name), "h_after_input_norm_%d.bin", layer);
            ref_t* input = load_at(inputs, name);
            bf16* x = narrow(input);
            for (int branch = 0; branch < 2; branch++) {
                char op = branch ? 'b' : 'a';
                snprintf(name, sizeof(name), "%c_%d.bin", op, layer);
                ref_t* expected = load_at(refs, name);
                bf16* y = calloc(expected->n_elem, sizeof(*y));
                snprintf(name, sizeof(name), "language_model.model.layers.%d.linear_attn.in_proj_%c.weight", layer, op);
                const st_tensor* w = st_find(archive, name);
                snprintf(name, sizeof(name), "language_model.model.layers.%d.linear_attn.in_proj_%c.scales", layer, op);
                const st_tensor* s = st_find(archive, name);
                snprintf(name, sizeof(name), "language_model.model.layers.%d.linear_attn.in_proj_%c.biases", layer, op);
                const st_tensor* b = st_find(archive, name);
                if (!w || !s || !b || !y) return 1;
                linear_q8_bf16(x, w->data, s->data, b->data, NULL, y,
                               (uint32_t)input->shape[0], (uint32_t)input->shape[1],
                               (uint32_t)expected->shape[1]);
                snprintf(name, sizeof(name), "L%d in_proj_%c", layer, op);
                fail |= err_check(y, expected->data, expected->n_elem, .03, name);
                free(y); ref_free(expected);
            }
            free(x); ref_free(input);
        }
        st_close(archive);
        return fail;
    }
    if (rms_all) {
        int fail = 0, checked = 0;
        for (int step = 0; step < 17; step++) {
            char input_dir[4096], reference_dir[4096];
            if (step) {
                snprintf(input_dir, sizeof(input_dir), "%s/decode-%03d", inputs, step);
                snprintf(reference_dir, sizeof(reference_dir), "%s/decode-%03d", refs, step);
            } else {
                snprintf(input_dir, sizeof(input_dir), "%s", inputs);
                snprintf(reference_dir, sizeof(reference_dir), "%s", refs);
            }
            for (int layer = 0; layer < 64; layer++) {
                for (int post = 0; post < 2; post++) {
                    char name[256];
                    snprintf(name, sizeof(name), "%s_%d.bin", post ? "x_after_attn_residual" : "x_in", layer);
                    ref_t* input = load_at(input_dir, name);
                    snprintf(name, sizeof(name), "%s_%d.bin", post ? "h_after_post_norm" : "h_after_input_norm", layer);
                    ref_t* expected = load_at(reference_dir, name);
                    snprintf(name, sizeof(name), "language_model.model.layers.%d.%s.weight", layer,
                             post ? "post_attention_layernorm" : "input_layernorm");
                    const st_tensor* weight = st_find(archive, name);
                    if (!weight) return 1;
                    bf16 *x = narrow(input), *y = calloc(input->n_elem, sizeof(*y));
                    rmsnorm_bf16(x, weight->data, y, (uint32_t)input->shape[0],
                                 (uint32_t)input->shape[1], 1e-6f);
                    char label[128];
                    snprintf(label, sizeof(label), "call%d L%d %s RMSNorm", step, layer, post ? "post" : "input");
                    fail |= err_check(y, expected->data, expected->n_elem, .01, label);
                    checked++;
                    free(x); free(y); ref_free(input); ref_free(expected);
                }
            }
        }
        printf("weighted RMSNorm replay: %d comparisons %s\n", checked, fail ? "FAIL" : "OK");
        st_close(archive);
        return fail;
    }
    if (rms_layer >= 0) {
        char name[256];
        snprintf(name, sizeof(name), "x_in_%d.bin", rms_layer);
        ref_t* input = load_at(inputs, name);
        snprintf(name, sizeof(name), "h_after_input_norm_%d.bin", rms_layer);
        ref_t* expected = load_at(refs, name);
        snprintf(name, sizeof(name), "language_model.model.layers.%d.input_layernorm.weight", rms_layer);
        const st_tensor* weight = st_find(archive, name);
        if (!weight) return 1;
        bf16 *x = narrow(input), *y = calloc(input->n_elem, sizeof(*y));
        rmsnorm_bf16(x, weight->data, y, (uint32_t)input->shape[0], (uint32_t)input->shape[1], 1e-6f);
        int fail = err_check(y, expected->data, expected->n_elem, .01, "same-input weighted RMSNorm");
        free(x); free(y); ref_free(input); ref_free(expected);
        st_close(archive);
        return fail;
    }
    ref_t *h = load_at(inputs, "h_after_input_norm_63.bin"), *want = load_at(refs, "v_proj_63.bin");
    bf16 *x = narrow(h), *y = calloc(want->n_elem, sizeof(*y));
    const st_tensor *w = st_find(archive, "language_model.model.layers.63.self_attn.v_proj.weight");
    const st_tensor *s = st_find(archive, "language_model.model.layers.63.self_attn.v_proj.scales");
    const st_tensor *b = st_find(archive, "language_model.model.layers.63.self_attn.v_proj.biases");
    if (!w || !s || !b || !y) return 1;
    linear_q8_bf16(x, w->data, s->data, b->data, NULL, y,
                   (uint32_t)h->shape[0], (uint32_t)h->shape[1], (uint32_t)want->shape[1]);
    int fail = err_check(y, want->data, want->n_elem, .03, "same-input v_proj L63");
    free(x); free(y); ref_free(h); ref_free(want);
    int layers[] = {3, 43, 51, 55, 59, 63};
    for (size_t i = 0; i < sizeof(layers)/sizeof(*layers); i++) {
        char name[128];
        int li = layers[i];
        snprintf(name, sizeof(name), "q_after_rope_%d.bin", li);
        ref_t* qr = load_at(inputs, name);
        snprintf(name, sizeof(name), "cache_k_%d.bin", li);
        ref_t* kr = load_at(inputs, name);
        snprintf(name, sizeof(name), "cache_v_%d.bin", li);
        ref_t* vr = load_at(inputs, name);
        snprintf(name, sizeof(name), "attn_%d.bin", li);
        want = load_at(refs, name);
        bf16 *q = narrow(qr), *k = narrow(kr), *v = narrow(vr);
        y = calloc(qr->n_elem, sizeof(*y));
        uint32_t lq = (uint32_t)qr->shape[0], lk = (uint32_t)kr->shape[0];
        uint32_t nq = (uint32_t)qr->shape[1], nk = (uint32_t)kr->shape[1];
        uint32_t d = (uint32_t)qr->shape[2];
        sdpa_gqa_bf16(q, k, v, y, lq, lk, nq, nk, d, lk-lq, 1.0f/sqrtf((float)d), 1);
        fail |= err_check(y, want->data, want->n_elem, .05, name);
        free(q); free(k); free(v); free(y);
        ref_free(qr); ref_free(kr); ref_free(vr); ref_free(want);
    }
    st_close(archive);
    return fail;
}
