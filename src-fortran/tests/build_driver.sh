#!/bin/sh
set -eu
compiler=$1
case "$compiler" in
    gfortran) flags="-std=f2018 -pedantic -Wall -Wextra -Werror -Waliasing -Wimplicit-interface -O2 -fcheck=all" ;;
    flang) flags="-std=f2018 -pedantic -Werror -O2" ;;
    *) echo "unsupported driver compiler" >&2; exit 1 ;;
esac
out=build/fortran-driver/$compiler
mkdir -p "$out"
"$compiler" --version > "$out/compiler.txt"
printf '%s\n' "$compiler $flags -fno-fast-math (compiler-default contraction)" > "$out/flags.txt"
sources="src-fortran/precision_formats.f90 src-fortran/dense_kernels.f90
    src-fortran/model_io_support.f90 src-fortran/json_tokens.f90
    src-fortran/safetensors_io.f90 src-fortran/qwen_config.f90 src-fortran/qwen_weights.f90
    src-fortran/inference_dump.f90 src-fortran/dense_model.f90"
"$compiler" $flags -fno-fast-math -J"$out" -I"$out" \
    $sources src-fortran/main.f90 -o "$out/qwen-raw"
"$compiler" $flags -fno-fast-math -J"$out" -I"$out" \
    $sources src-fortran/tests/test_driver_api.f90 -o "$out/test_driver_api"
"$compiler" $flags -fno-fast-math -J"$out" -I"$out" \
    src-fortran/precision_formats.f90 src-fortran/dense_kernels.f90 \
    src-fortran/tests/native_replay.f90 -o "$out/native-replay"
python - "$out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = dict(sources={str(p): sha(p) for p in Path("src-fortran").glob("*.f90")},
                binary_sha256=sha(out/"qwen-raw"), compiler_flags=(out/"flags.txt").read_text())
(out/"build-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
PY
