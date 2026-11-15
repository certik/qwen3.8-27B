#ifndef COMPARE_H
#define COMPARE_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../utils/bf16.h"

/* A status, never an error magnitude. Non-finite oracle values are invalid. */
static int check_f32(const float* got, const float* want, size_t n,
                     double atol, double rtol, const char* label) {
    double max_d = 0;
    size_t first = n;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)got[i] - want[i]);
        if (d > max_d) max_d = d;
        if ((!isfinite(got[i]) || !isfinite(want[i]) ||
             d > atol + rtol * fabs(want[i])) && first == n) first = i;
    }
    int fail = n == 0 || first != n;
    printf("  %-32s max|d|=%.4g (n=%zu) %s\n", label, max_d, n,
           fail ? "FAIL" : "OK");
    if (first < n)
        printf("    first FAIL at %zu: got=%g want=%g\n",
               first, got[first], want[first]);
    return fail;
}

static int err_check(const bf16* got, const float* want, size_t n,
                     double tol, const char* label) {
    float* f = malloc(n * sizeof(*f));
    if (!f && n) return 1;
    for (size_t i = 0; i < n; ++i) f[i] = bf16_to_f32(got[i]);
    int fail = check_f32(f, want, n, tol, tol, label);
    free(f);
    return fail;
}
#endif
