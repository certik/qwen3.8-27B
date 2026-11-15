#!/bin/sh
set -eu
compiler=$1
case "$compiler" in
    gfortran) flags="-std=f2018 -pedantic -Wall -Wextra -Werror -Wimplicit-interface -O2 -fcheck=all" ;;
    flang) flags="-std=f2018 -pedantic -Werror -O2" ;;
    *) echo "unsupported tokenizer test compiler" >&2; exit 1 ;;
esac
out=build/fortran-tokenizer/$compiler
mkdir -p "$out"
"$compiler" --version > "$out/compiler.txt"
"$compiler" $flags -J"$out" -I"$out" \
    src-fortran/qwen_tokenizer.f90 src-fortran/tests/test_tokenizer.f90 \
    -o "$out/test_tokenizer"
