// tests/load_ref.h — tiny loader for the .bin format produced by
// tools/dump_ref.py.  Drop into tests/.

#ifndef LOAD_REF_H
#define LOAD_REF_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define MAGIC_LEN 8
static const char REF_MAGIC[MAGIC_LEN+1] = "LLMTNSR1";

typedef struct {
    uint32_t dtype;        // 0=f32, 1=i32
    uint32_t ndim;
    int64_t  shape[6];
    size_t   n_elem;
    void*    data;
} ref_t;

static const char* ref_directory = NULL;

static ref_t* ref_load(const char* path) {
    char joined[4096];
    if (ref_directory) {
        const char* name = strrchr(path, '/');
        int n = snprintf(joined, sizeof(joined), "%s/%s", ref_directory,
                         name ? name + 1 : path);
        if (n < 0 || (size_t)n >= sizeof(joined)) {
            fprintf(stderr, "reference path too long\n"); exit(1);
        }
        path = joined;
    }
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[MAGIC_LEN];
    if (fread(magic, 1, MAGIC_LEN, f) != MAGIC_LEN ||
        memcmp(magic, REF_MAGIC, MAGIC_LEN) != 0) {
        fprintf(stderr, "%s: bad magic\n", path); exit(1);
    }
    ref_t* r = calloc(1, sizeof(*r));
    if (!r || fread(&r->dtype, 4, 1, f) != 1 ||
        fread(&r->ndim, 4, 1, f) != 1 ||
        r->dtype > 1 || r->ndim > 6) {
        fprintf(stderr, "%s: invalid header\n", path); exit(1);
    }
    size_t n = 1;
    for (uint32_t i = 0; i < r->ndim; i++) {
        if (fread(&r->shape[i], 8, 1, f) != 1 || r->shape[i] <= 0 ||
            (uint64_t)r->shape[i] > SIZE_MAX / 4 / n) {
            fprintf(stderr, "%s: invalid shape\n", path); exit(1);
        }
        n *= (size_t)r->shape[i];
    }
    r->n_elem = n;
    size_t elem = (r->dtype == 0) ? 4 : 4;
    r->data = malloc(n * elem);
    if (!r->data || fread(r->data, elem, n, f) != n || fgetc(f) != EOF) {
        fprintf(stderr, "%s: invalid payload\n", path); exit(1);
    }
    fclose(f);
    return r;
}

static void ref_free(ref_t* r) { if (r) { free(r->data); free(r); } }

// Compare an f32 reference against a bf16 buffer; print max-abs-error.
// Returns 0 on success (<= tol), nonzero otherwise.
#include "../utils/bf16.h"
static int ref_check_bf16_vs_f32(const bf16* got, const float* want, size_t n,
                                 float tol, const char* label) {
    float max_d = 0.0f;
    size_t bad_at = (size_t)-1;
    float bad_got = 0, bad_want = 0;
    for (size_t i = 0; i < n; i++) {
        float g = bf16_to_f32(got[i]);
        float w = want[i];
        if (!isfinite(g) || !isfinite(w)) return 1;
        float d = g > w ? g - w : w - g;
        if (d > max_d) { max_d = d; bad_at = i; bad_got = g; bad_want = w; }
    }
    printf("  %-30s max|d|=%g  (n=%zu)\n", label, max_d, n);
    if (max_d > tol) {
        printf("    first bad at %zu: got=%g want=%g\n", bad_at, bad_got, bad_want);
        return 1;
    }
    return 0;
}

#endif
