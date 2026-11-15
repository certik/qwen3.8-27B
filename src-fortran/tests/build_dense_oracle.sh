#!/bin/sh
set -eux
printf '%s\n' \
    '694695f941bde8197f8efc82ff5cf40089dab4975c8973c9a663f55c02c5857e  src-cpu/kernels.c' \
    '4ee4215889b773293f9343470e51b4404ccb236f1b839c2ccb34a5c4a096dc4f  src-cpu/kernels.h' \
    | shasum -a 256 -c -
mkdir -p build/fortran-dense
# The immutable reference has an unused q8_deq helper; retain its diagnostic.
if command -v xcrun >/dev/null 2>&1; then
    set -- xcrun clang
else
    set -- ${CC:-clang}
fi
"$@" -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    -Wno-error=unused-function -D_DEFAULT_SOURCE -fno-fast-math -ffp-contract=off \
    -fno-vectorize -fno-slp-vectorize -Isrc-cpu \
    -c src-cpu/kernels.c -o build/fortran-dense/kernels.o
"$@" -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
    -D_DEFAULT_SOURCE -fno-fast-math -ffp-contract=off -fno-vectorize -fno-slp-vectorize \
    -Isrc-cpu src-fortran/tests/dense_oracle.c build/fortran-dense/kernels.o \
    -lm -o build/fortran-dense/dense_oracle
