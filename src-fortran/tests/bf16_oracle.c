#include <fenv.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bf16.h"

#pragma STDC FENV_ACCESS ON

enum { record_count = 2 * 256 * 7 * 5 + 4096 + 16 };

static void emit(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    const bf16 expected = f32_to_bf16(value);
    if (printf("%08" PRIx32 " %04" PRIx16 "\n", bits, expected) < 0) {
        fprintf(stderr, "cannot write oracle record\n");
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    const uint32_t fractions[] = {0, 1, 2, 63, 64, 126, 127};
    const uint32_t tails[] = {0, 0x7fff, 0x8000, 0x8001, 0xffff};
    const uint32_t specials[] = {
        0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7f800001, 0xff800001, 0x7fc00001, 0xffc00001
    };
    float one = 1.0f;
    uint32_t one_bits = 0;
    if (CHAR_BIT != 8 || sizeof(float) != 4 || sizeof(uint32_t) != 4 ||
        sizeof(bf16) != 2 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 ||
        FLT_MIN_EXP != -125 || FLT_MAX_EXP != 128) {
        fprintf(stderr, "C oracle requires IEEE binary32 and exact-width integers\n");
        return EXIT_FAILURE;
    }
    memcpy(&one_bits, &one, sizeof(one_bits));
    if (one_bits != UINT32_C(0x3f800000) || feclearexcept(FE_ALL_EXCEPT) != 0) {
        fprintf(stderr, "C oracle representation/floating environment mismatch\n");
        return EXIT_FAILURE;
    }
    if (printf("BF16_ORACLE_V1 %d\n", record_count) < 0) return EXIT_FAILURE;
    for (uint32_t sign = 0; sign < 2; ++sign) {
        for (uint32_t exponent = 0; exponent < 256; ++exponent) {
            for (size_t f = 0; f < sizeof(fractions) / sizeof(fractions[0]); ++f) {
                for (size_t t = 0; t < sizeof(tails) / sizeof(tails[0]); ++t) {
                    emit((sign << 31) | (exponent << 23) | (fractions[f] << 16) | tails[t]);
                }
            }
        }
    }
    uint32_t random_bits = UINT32_C(0x6d2b79f5);
    for (unsigned i = 0; i < 4096; ++i) {
        random_bits ^= random_bits << 13;
        random_bits ^= random_bits >> 17;
        random_bits ^= random_bits << 5;
        emit(random_bits);
    }
    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); ++i) emit(specials[i]);
    if (fetestexcept(FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW) != 0) {
        fprintf(stderr, "C bit-only helper unexpectedly raised an IEEE exception\n");
        return EXIT_FAILURE;
    }
    if (printf("END %d\n", record_count) < 0 || fflush(stdout) != 0) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
