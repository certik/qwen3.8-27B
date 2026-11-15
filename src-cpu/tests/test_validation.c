#include <stdlib.h>
#include "compare.h"
#include "load_ref.h"
#include "../kernels.h"

static int precision_boundaries(void) {
    bf16 x[] = {0x3eab, 0x3f8d, 0xbfa7, 0x4003};
    bf16 w[] = {0x3f91, 0x3f25, 0x3fc7, 0x3ef3};
    bf16 y[4], gated[4];
    float sq = 0;
    for (int i = 0; i < 4; i++) sq += bf16_to_f32(x[i]) * bf16_to_f32(x[i]);
    float inv = 1.0f / sqrtf(sq / 4 + 1e-6f);
    rmsnorm_bf16(x, w, y, 1, 4, 1e-6f);
    int fail = 0;
    for (int i = 0; i < 4; i++) {
        bf16 normalized = f32_to_bf16(bf16_to_f32(x[i]) * inv);
        fail |= y[i] != f32_to_bf16(bf16_to_f32(normalized) * bf16_to_f32(w[i]));
    }
    // Enumerate BF16 gate inputs: FFN and RMSNormGated have different stores.
    int distinct = 0;
    for (unsigned bits = 0x3c00; bits < 0x4100; bits++) {
        bf16 gate = (bf16)bits, up = 0x3f93, out;
        float g = bf16_to_f32(gate);
        float e = bf16_to_f32(f32_to_bf16(expf(g)));
        float reciprocal = bf16_to_f32(f32_to_bf16(1 / bf16_to_f32(f32_to_bf16(1+e))));
        float sigmoid = bf16_to_f32(f32_to_bf16(1 - reciprocal));
        float s = g * sigmoid;
        silu_mul_bf16(&gate, &up, &out, 1);
        bf16 expected = f32_to_bf16(bf16_to_f32(f32_to_bf16(s)) * bf16_to_f32(up));
        fail |= out != expected;
        distinct += expected != f32_to_bf16(s * bf16_to_f32(up));
    }
    silu_mul_f32_gate_bf16(x, w, gated, 4);
    for (int i = 0; i < 4; i++) {
        float g = bf16_to_f32(x[i]);
        fail |= gated[i] != f32_to_bf16((g / (1 + expf(-g))) * bf16_to_f32(w[i]));
    }
    fail |= distinct == 0;
    bf16 a[] = {0x3eab, 0xc300}, al[] = {0, 0}, dt[] = {0x3e01, 0};
    float decay[2];
    compute_g_f32(a, al, dt, decay, 1, 2);
    float sum = bf16_to_f32(f32_to_bf16(bf16_to_f32(a[0]) + bf16_to_f32(dt[0])));
    float e = bf16_to_f32(f32_to_bf16(expf(-fabsf(sum))));
    float lp = bf16_to_f32(f32_to_bf16(log1pf(e)));
    float sp = bf16_to_f32(f32_to_bf16(fmaxf(sum, 0) + lp));
    fail |= fabsf(decay[0] - expf(-sp)) > 1e-6f || decay[1] != 1.0f;
    bf16 near_zero_a = 0x4051, near_zero_dt = 0xc054, native_al = 0xc06d;
    compute_g_f32(&near_zero_a, &native_al, &near_zero_dt, decay, 1, 1);
    // Frozen native GPU result: a=3.265625, dt=-3.3125, A_log=-3.703125.
    fail |= fabsf(decay[0] - 0.9836717844009399f) > 1e-7f;
    // A non-BF16-representable FP32 gate must survive consecutive calls.
    bf16 zero = 0, q = 0x3f80, value = 0, result;
    float state = 3.25f, gate = 0.99713f;
    for (int i = 0; i < 80; i++)
        gated_delta_step_bf16(&q, &zero, &value, &gate, &zero, &state, &result, 1, 1, 1, 1);
    float expected = 3.25f;
    for (int i = 0; i < 80; i++) expected *= gate;
    fail |= state != expected;
    bf16 router[] = {0xbef5, 0xbda1, 0xbe90, 0xbe90};
    int32_t experts[2];
    float weights[2];
    softmax_topk_bf16(router, experts, weights, 1, 4, 2, 1);
    fail |= !((experts[0] == 1 && experts[1] == 3) ||
              (experts[0] == 3 && experts[1] == 1));
    bf16 expert_values[8*64], mixed[64];
    float expert_weights[8];
    for (int k = 0; k < 8; k++) {
        expert_weights[k] = .125f;
        for (int d = 0; d < 64; d++) expert_values[k*64+d] = f32_to_bf16(k ? .03125f : 8.0f);
    }
    expert_mix_bf16(expert_values, expert_weights, mixed, 1, 8, 64);
    for (int d = 0; d < 64; d++) fail |= mixed[d] != f32_to_bf16(1.0f);
    printf("precision boundaries: %s\n", fail ? "FAIL" : "OK");
    return fail;
}

int main(int argc, char** argv) {
    if (argc == 2) {
        ref_t* r = ref_load(argv[1]);
        ref_free(r);
        return 0;
    }
    bf16 good[] = {f32_to_bf16(1.0f)};
    bf16 bad[] = {f32_to_bf16(2.0f)};
    bf16 nan[] = {0x7fc0};
    bf16 inf[] = {0x7f80};
    float want[] = {1};
    float invalid[] = {NAN};
    int fail = err_check(good, want, 1, .01, "matching") != 0;
    fail |= err_check(bad, want, 1, .01, "mismatch") != 1;
    fail |= err_check(nan, want, 1, .01, "NaN actual") != 1;
    fail |= err_check(inf, want, 1, .01, "infinite actual") != 1;
    fail |= err_check(good, invalid, 1, .01, "NaN oracle") != 1;
    fail |= ref_check_bf16_vs_f32(nan, want, 1, .01, "legacy NaN") != 1;
    fail |= check_f32(want, invalid, 1, 1e-5, 1e-5, "FP32 state NaN") != 1;
    return fail | precision_boundaries();
}
