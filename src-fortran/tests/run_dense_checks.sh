#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: run_dense_checks.sh executable oracle.bin output-directory" >&2
    exit 2
fi
executable=$1
oracle=$2
output=$3
mkdir -p "$output"

status=0
"$executable" "$oracle" > "$output/positive.log" 2>&1 || status=$?
cat "$output/positive.log"
if [ "$status" -ne 0 ]; then exit "$status"; fi

reject() {
    name=$1
    argument=$2
    diagnostic=$3
    status=0
    "$executable" "$argument" > "$output/$name.log" 2>&1 || status=$?
    if [ "$status" -eq 0 ] || [ "$status" -ge 128 ]; then
        cat "$output/$name.log" >&2
        echo "FAIL: $name returned $status, expected a controlled nonzero exit" >&2
        exit 1
    fi
    if ! grep -F "$diagnostic" "$output/$name.log" > /dev/null; then
        cat "$output/$name.log" >&2
        echo "FAIL: $name did not emit the expected diagnostic" >&2
        exit 1
    fi
    printf 'PASS: reject %s (exit %s): %s\n' "$name" "$status" "$diagnostic"
}

reject id-low --bad-id-low "embedding token id outside 0..vocabulary-1"
reject id-high --bad-id-high "embedding token id outside 0..vocabulary-1"
reject q8-group --bad-q8-group "q8 requires a positive multiple of 64 input features"
reject q8-shape --bad-q8-shape "q8 packed weight, scale or offset shape mismatch"
reject linear-output --bad-linear-output "linear output shape mismatch"
reject bias --bad-bias "linear bias shape mismatch"
reject norm-output --bad-norm-output "rmsnorm output shape mismatch"
reject norm-weight --bad-norm-weight "rmsnorm weight shape mismatch"
reject epsilon --bad-epsilon "rmsnorm requires positive epsilon"
reject element --bad-element "elementwise shape mismatch"
reject residual --bad-residual "residual shape mismatch"
reject empty-argmax --empty-argmax "dense_kernels requires nonempty arrays"
reject empty-silu --empty-silu "dense_kernels requires nonempty arrays"
reject g-parameter --bad-g-parameter "compute_g shape mismatch"
reject g-output --bad-g-output "compute_g shape mismatch"
reject empty-g --empty-g "dense_kernels requires nonempty arrays"
reject conv-carry --bad-conv-carry "causal convolution shape mismatch"
reject conv-weights --bad-conv-weights "causal convolution shape mismatch"
reject conv-output --bad-conv-output "causal convolution shape mismatch"
reject empty-conv-kernel --empty-conv-kernel "dense_kernels requires nonempty arrays"
reject delta-ratio --bad-delta-ratio "gated delta requires Hv divisible by Hk"
reject delta-k --bad-delta-k "gated delta shape mismatch"
reject delta-state --bad-delta-state "gated delta shape mismatch"
reject delta-output --bad-delta-output "gated delta shape mismatch"
reject delta-g --bad-delta-g "gated delta shape mismatch"
reject delta-beta --bad-delta-beta "gated delta shape mismatch"
reject empty-delta-q --empty-delta-q "dense_kernels requires nonempty arrays"
reject odd-rope --odd-rope "rope requires even rotation width in 0..D"
reject wide-rope --wide-rope "rope requires even rotation width in 0..D"
reject negative-rope --negative-rope "rope requires even rotation width in 0..D"
reject rope-frequencies --bad-rope-frequencies "rope frequency shape mismatch"
reject negative-position --negative-position "position offset outside uint32 range"
reject wide-position --wide-position "position offset outside uint32 range"
reject overflow-position --overflow-position "position range exceeds uint32"
reject sdpa-ratio --bad-sdpa-ratio "sdpa requires Nq divisible by Nkv"
reject sdpa-head-width --bad-sdpa-head-width "sdpa shape mismatch"
reject sdpa-v --bad-sdpa-v "sdpa shape mismatch"
reject sdpa-output --bad-sdpa-output "sdpa shape mismatch"
reject empty-sdpa-cache --empty-sdpa-cache "dense_kernels requires nonempty arrays"
reject sdpa-position --bad-sdpa-position "position offset outside uint32 range"
reject sdpa-overflow-position --overflow-sdpa-position "position range exceeds uint32"
printf 'not a native fixture\n' > "$output/bad.fixture"
reject bad-oracle "$output/bad.fixture" "dense C oracle native representation mismatch"
rm -f "$output/missing.fixture"
reject missing-oracle "$output/missing.fixture" "cannot open dense C oracle"
