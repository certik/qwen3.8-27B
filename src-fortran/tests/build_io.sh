#!/bin/sh
set -eu
compiler=$1
case "$compiler" in
    gfortran) flags="-std=f2018 -pedantic -Wall -Wextra -Werror -Wimplicit-interface -O2 -fcheck=all" ;;
    flang) flags="-std=f2018 -pedantic -Werror -O2" ;;
    *) echo "unsupported I/O test compiler" >&2; exit 1 ;;
esac
out=build/fortran-io/$compiler
mkdir -p "$out"
"$compiler" --version > "$out/compiler.txt"
"$compiler" $flags -J"$out" -I"$out" \
    src-fortran/precision_formats.f90 \
    src-fortran/model_io_support.f90 src-fortran/json_tokens.f90 \
    src-fortran/safetensors_io.f90 src-fortran/qwen_config.f90 \
    src-fortran/qwen_weights.f90 src-fortran/tests/test_model_io.f90 \
    -o "$out/test_model_io"
