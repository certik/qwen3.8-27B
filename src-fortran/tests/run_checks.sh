#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: run_checks.sh executable oracle.txt output-directory" >&2
    exit 2
fi
executable=$1
oracle=$2
output=$3
mkdir -p "$output"

status=0
"$executable" "$oracle" > "$output/positive.log" 2>&1 || status=$?
cat "$output/positive.log"
if [ "$status" -ne 0 ]; then
    exit "$status"
fi

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

replace_first_record() {
    # Preserve the remaining full corpus, so a mismatch cannot pass as truncation.
    {
        IFS= read -r header
        IFS= read -r first_record
        printf '%s\n%s\n' "$header" "$1"
        cat
    } < "$oracle" > "$2"
}

rm -f "$output/missing.oracle"
{
    IFS= read -r header
    IFS= read -r first_record
    printf '%s\n%s\n' "$header" "$first_record"
} < "$oracle" > "$output/truncated.oracle"
replace_first_record "00000000 0001" "$output/corrupted.oracle"
replace_first_record "00000000 000g" "$output/nonhex.oracle"
replace_first_record "00000000 7fc0" "$output/unexpected-nan.oracle"
replace_first_record "00000000 7f80" "$output/unexpected-infinity.oracle"
{ cat "$oracle"; printf 'EXTRA\n'; } > "$output/extra.oracle"

reject invalid-low --invalid-low "packed_q8_byte requires byte_index in 0..3"
reject invalid-high --invalid-high "packed_q8_byte requires byte_index in 0..3"
reject missing "$output/missing.oracle" "cannot open C oracle output"
reject truncated "$output/truncated.oracle" "missing/truncated C oracle output"
reject corrupted "$output/corrupted.oracle" "C oracle mismatch at record 1"
reject nonhex "$output/nonhex.oracle" "non-hex C oracle record"
reject unexpected-nan "$output/unexpected-nan.oracle" "unexpected NaN classification in C oracle"
reject unexpected-infinity "$output/unexpected-infinity.oracle" "unexpected finite-to-infinity conversion in C oracle"
reject extra "$output/extra.oracle" "unexpected extra data or I/O error after C oracle footer"
