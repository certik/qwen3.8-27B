#!/bin/sh
set -eu
# These hashes are the dense CPU reference this driver is checked against
# (integration commit 97534c6, math unchanged from 37ff651). README text is
# not part of the pin.
printf '%s\n' \
    'da550f8a0b4e391d6c22899b4a486f58faa1f223ca3ac9279b23f85e565c2d70  src-cpu/main.c' \
    '694695f941bde8197f8efc82ff5cf40089dab4975c8973c9a663f55c02c5857e  src-cpu/kernels.c' \
    '4ee4215889b773293f9343470e51b4404ccb236f1b839c2ccb34a5c4a096dc4f  src-cpu/kernels.h' \
    'ee8dc1119710d9654fa69663d8cb7e176be68d1efe64a0924f4bd16c0f646114  src-cpu/safetensors.c' \
    '5592dcd58cf255aedc0e8d5dcb8c884da16589634cebc18b55e534171793b492  src-cpu/tokenizer.c' \
    '08e2e732754233912aad0fd33c135b9bf90838ef3160876b3f51763cfb9d94ee  src-cpu/qwen_chat.c' \
    '7b052ed56b3d0994c8740e6977f9c690508ac1be8621b84a2d6d31ddc693284d  src-cpu/utils/json.c' \
    '7448f188f75e2fdbbce72d10d5a5f44dc5324bb9fe3d836ed7e4ada7021467f3  src-cpu/utils/iofile.c' \
    '07eff9558235a63e222d9ad9ea4fcf9c5662fb093982cf81c284f243bb12736d  src-cpu/utils/bytebuf.c' \
    '995e62eb164c81e72ed76155a1225b459753085e17d0201fa61c77329240823c  src-cpu/utils/utf8.c' \
    '87776eb4eb4f4c6b6d5c3517298f7ba5cc73744a2597ea017141854b405beb16  src-cpu/utils/bf16.h' \
    | shasum -a 256 -c -
mkdir -p build/fortran-driver
{
    printf 'C reference pinned by sha256 in this script (formerly 97534c6 / math 37ff651)\n'
    shasum -a 256 src-cpu/main.c src-cpu/kernels.c src-cpu/kernels.h src-cpu/safetensors.c \
        src-cpu/tokenizer.c src-cpu/qwen_chat.c src-cpu/utils/*.c src-cpu/utils/bf16.h
} > build/fortran-driver/source-pins.txt
if command -v xcrun >/dev/null 2>&1; then
    set -- xcrun clang
else
    set -- ${CC:-clang}
fi
"$@" -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
    -D_DEFAULT_SOURCE -fno-fast-math -ffp-contract=off -fno-vectorize -fno-slp-vectorize -Isrc-cpu \
    src-cpu/main.c src-cpu/kernels.c src-cpu/safetensors.c src-cpu/tokenizer.c src-cpu/qwen_chat.c \
    src-cpu/utils/json.c src-cpu/utils/iofile.c src-cpu/utils/bytebuf.c src-cpu/utils/utf8.c \
    -lm -o build/fortran-driver/qwen-c-oracle
