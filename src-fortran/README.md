# Pure-Fortran raw-ID dense-hybrid reference

Text prompts, the chat template and text output are handled by `qwen_tokenizer.f90` and `main.f90`
(see the top-level README); the id-only interface below is unchanged.

## Current FN native-semantic CPU profile

This branch replaces the obsolete scalar-C numerical graph with standard
Fortran math suitable for subsequent generic compiler offload. **CPU only;
no offload or speed claim.** Model/I/O/math do not link C, MLX, OpenMP, BLAS,
or vendor intrinsics. BF16 remains `integer(int16)` bit storage, packed
unsigned affine q8/group64 remains `integer(int32)`, and accumulation and
recurrent state remain `real(real32)`. Independent work uses legal
`DO CONCURRENT`, construct-local indices and `BLOCK` temporaries. FP32
contractions use `SUM`; time and model layers remain sequential.

Decoded projection q8 codes use numeric `integer(int16)` values 0..255,
distinct from BF16 bit storage. Each arithmetic consumer converts codes with
`real(..., real32)` at use; no decoded FP32 code array is retained. Packed
model words, selectors, row reuse, FP32 reductions and all BF16 rounding
boundaries are unchanged.

The preserved scalar-profile checkout at `a70b69e` and its fixtures are
independent diagnostic history. Native success does not repair any historical
C/Metal FAIL or outlier. **The documentation after the historical divider
below describes that older profile, not FN's current numerical/generation
contract.** Model loading and representation APIs remain as described in
[MODEL_IO.md](MODEL_IO.md) and [CONTRACT.txt](CONTRACT.txt).

### Source-to-native family map

Reference: read-only `mlx-native`, mathematical source
`8640d2dfd8404b803b8621ef2b80a78ee0bfc0dc`; `9dd05ab` adds measurement only.
Underlying MLX `68cf2fddd8de5edd8ab3d926391772b2e2cedad8`, MLX-LM
`ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`. Paths below are relative to
`src-metal/native` unless otherwise stated. This is a semantic port, **not**
a literal Metal-body or device-instruction port.

| Family | Pinned source / current Fortran behavior |
| --- | --- |
| qmv | `vendor/mlx/backend/metal/kernels/quantized.h`, `native_ops.c:native_qmv_eligible`; groupwise `scale*SUM(code*x)+offset*SUM(x)`, FP32, final BF16. Width/output alignment and M10/12/18 thresholds are checked by the production helper. |
| qmm | `templates/qmm.metal.in`, `native_ops.c:native_matrix_parts`; BF16-staged dequantized weights, FP32 contractions, BF16 split partials and typed BF16 balanced reduction. Native split shape policy retained; no QMM64 exact-shape specialization. |
| RMS | `vendor/.../rms_norm.metal`; FP32 sum of squares, normalize-before-weight BF16 store. Unweighted/scaled forms preserve their separate BF16 stores and epsilon. No Kahan sum. |
| Elementwise / decay | `templates/{conv_silu,dense_swiglu,gated_output,decay_strided_*}.metal.in`, `vendor/.../{unary_ops,binary_ops,bf16_math}.h`; typed BF16 sigmoid/softplus stages; gated RMS output uses stable FP32 sigmoid/SiLU; `g=exp(-exp(A_log)*softplus_BF16)` is multiplicative FP32. |
| Convolution / recurrence | `templates/depthwise_conv.metal.in`, pinned MLX-LM `gated_delta.py`; causal carry and FP32 state, sequential time, independent head/value rows, rank-one update after the old decayed-state contraction. No second exp in recurrence. |
| RoPE | `vendor/.../rope.metal`; split-half rotary64, log2/exp2-family inverse frequencies, FP32 angles/trig, BF16 outputs and untouched tail. CPU libm is not asserted bit-identical to Metal fast trig. |
| GQA attention | `native_math.c:native_attention`, `vendor/.../{sdpa_vector,softmax}.h`; M×GQA eligibility, FP32 vector attention, strided two-pass BF16 partials with FP32 sums/maxima at Lk1024+, generic BF16 Q-scale/QK/softmax/PV graph otherwise. No invented flash-attention route. |
| Sampler | `native_ops.c:native_sampler_dispatch`, `vendor/.../logsumexp.h`; FP32 logsumexp reduction, BF16 scalar, BF16 subtraction, then **first-index argmax**. `argmax_bf16` alone is not the model sampler. |
| Demand / generation | `main.c:{forward_demanded,generate_native}`; cache-only P−1 prefix in chunks≤2048, terminal-layer suffix pruning, final prompt token, one final unemitted forward. CPU execution is synchronous but performs the same logical work. |

Device SIMD association is deliberately not encoded in portable source.
BF16 roundpoints and state types are preserved; internal exact bytes, numerical
tolerance and final IDs are **different gates**. Saved native qmm/recurrent
receipts and unchanged C bounds are tested separately. CPU internal differences
are recorded with complete outputs, not dismissed as trajectory “drift.”

### Current API / protocol changes

`forward_ids(model,ids,logits,directory[,cache_only])` now returns
`logits(vocab,1)` for the final input row, or `logits(vocab,0)` for cache-only
demand. It no longer computes a whole-vocabulary head for every prefix row.
`sample_logits(model,logits,index[,logprobs])` calls the rounded native sampler
and records its actual invocation. `model%counts` records actual forwards,
cache-only forwards, heads, samplers, processed cache tokens and terminal
prefixes pruned; reset clears state **and** these per-request counts.

The raw-ID CLI emits schema3 and `native-semantic-fortran-cpu`, not the old
`mlx-0.31.2-m4max`/compensated-FP32 labels. For P>1/G65 it performs 67 forwards,
one cache-prefix, 66 heads/samplers, 65 consumed cached steps, 64 emitted cached
steps, and P+G logical cache tokens. P1 omits only the prefix. EOS still consumes
one lookahead input. `--teacher-ids` requires G IDs (including lookahead), not
G−1. `--reset-check` reruns the **whole request** and checks every emitted and
consumed ID, final lookahead, complete recurrent/conv/KV state (including spare
capacity), final logits and logprobs bit-for-bit.

`--dump-layer N` selects one complete layer plus embedding/final outputs.
All debug dumps have an enforced **256 MiB total budget**; directories must
exist. No real-model all-tensor trace is required or recommended.

### Reproducible CPU validation

From FN, through its provisioned CPU-only Pixi environment:

```sh
pixi run -e fortran-precision precision-versions
pixi run -e fortran-precision native-cpu-test > artifacts/fortran-native/native-tests.log 2>&1
pixi run -e fortran-precision native-cpu-replay > artifacts/fortran-native/native-replay.log 2>&1
pixi run -e fortran-precision precision-test > artifacts/fortran-native/precision.log 2>&1
pixi run -e fortran-precision io-test > artifacts/fortran-native/io.log 2>&1
```

`native-cpu-test` builds with approved PATH Flang24/GFortran16 through Pixi,
Fortran2018/pedantic/O2/no-fast-math, with GFortran runtime checks. Computational
binaries have no C oracle linkage. The independent conversion/C oracle tasks
remain test-only. `driver-test`, `dense-test`, and `attention-state-test` select
the current profile. The old `test_dense_kernels.f90`/`test_driver.py` exact
scalar-profile fixtures remain unmodified historical tests, not FN gates.

Model-free coverage: signed/nonzero affine metadata, packed high bits,
BF16 midpoints, aligned qmv and matrix fallback, qmm splits/tails, all helpers,
nondefault RMS epsilon, short convolution carry, GQA M5/M6 and Lk1023/1024/1025,
full saved native attention/RoPE/embedding/qmm/recurrent outputs, finiteness,
unchanged C output/state bounds, and tiny whole graphs P1/P5/P6/P18 with full
reset. Saved evidence is read-only; no GPU process or native library is loaded.
Compact, non-overwritten per-run receipts are under ignored
`artifacts/fortran-native/validation-*`. The original C linear bound is
0.03 absolute + 0.03 relative; recurrent output/state bounds remain
0.05/0.05 and 1e-5/1e-5 respectively.

Additional controls exercise P18/G65's actual 67/1/66/65/64 protocol,
P2049/P2050 prefix chunk boundaries on the tiny model, full-request teacher
forcing including lookahead, insufficient-teacher rejection, early-EOS
lookahead/reset, and exact signed-zero/NaN-payload state comparison.
`native-cpu-admission-tests` exercises successful and failing **mock**
admission, missing/malformed authorization, one-use lease ownership and
first-ID-divergence receipts. Its explicitly labeled unit receipts do not
launch a model and are not real-model evidence.

```sh
pixi run -e fortran-precision native-model-check --preflight-only
# Only after the parent provides a CPU model lease:
pixi run -e fortran-precision native-model-check --compiler flang \
    --lease artifacts/fortran-native/cpu-model-lease.json
```

The real-model helper reuses pinned input/native-ID and model identity
receipts, checks small metadata/header hashes and unchanged shard stats
(not another whole-weight hash/copy), validates the compiler flags and
source-to-binary build manifest, and requires AC/memory/disk preflight,
including ≥55% reported free memory at entry as a conservative admission
signal, not a measurement of physical residency. Missing/invalid RSS for a
live child is a fatal monitor error; an observed child-exit race is distinct.
Each lease is consumed once; the helper owns one CPU child with a 38 GiB
RSS ceiling and 1800-second timeout, retains compact completion/failure
receipts, and always releases its child and lock. No GPU or timing gate
is part of this phase.

### Real27B CPU acceptance and subsequent diagnostic correction

The one-use, no-dump controls ran at source **`98773cd4c02bb619573e26166919661627fee382`**:

| Control | Status | Supervised CPU cost | Peak RSS |
| --- | --- | --- | --- |
| Flang24 short18/17 + complete reset | **PASS** | 1663.13 s | 26.29 GiB |
| GFortran16 conditional short18/17 + reset | **TIMEOUT** at the unchanged 1800 s guard; no result returned | 1800.30 s | 26.96 GiB sampled |

These are costs, **not speed acceptance measurements**. They include model
initialization, inference, full-request reset/replay, monitoring and teardown;
the metadata preflight/Pixi activation is outside this timer. Both entries
reported 89% free memory, passed AC/identity/disk checks, and stayed below
38 GiB RSS. Both processes exited and their leases were consumed exactly once.

Flang emitted the complete pinned native sequence
`[9419,0,353,1044,1167,16451,11,264,3349,3992,1558,27222,7633,539,52540,5559,579]`
and lookahead `48696`. Actual per-request counts were 19 forwards, one
cache-only prefix, 18 heads/samplers, 17 consumed cached steps, 16 emitted
cached steps and 35 logical cache tokens. All emitted/consumed IDs, final
lookahead, complete cache/state, logits and logprobs matched the reset replay
bit-for-bit (38 forwards/70 processed tokens including reset). The implemented
per-forward logits and persistent FP32 recurrent-state finite checks passed;
no all-intermediate-tensor trace or universal native-byte equality is claimed.

GFortran's last sample exceeded only the time bound (not RSS). It returned
no result/IDs/reset receipt, so those functional gates are **NOT_COMPLETE**,
not a demonstrated numerical mismatch. It was not retried.

Receipts are under `artifacts/fortran-native/model-flang-1793568632043775000`
and `model-gfortran-1793570457713080000`. Exact pre-diagnostic executables
and build manifests are retained in `pre-diagnostic-binaries/`; SHA-256:
Flang `e6298c496bed9c8283e8951c4f9785d3c7141402fbe6e5d0850e5a2011665b34`,
GFortran `d705234ba121c592463b365fb77b5c68c8aaae8c59fcf440763870a526b21012`.

After both controls exited, a diagnostic-only guard was added: a terminal
recurrent cache-only prefix no longer dumps stale `z` when its producer is
pruned. A five-layer/attention-interval4 regression failed before the guard
and passes under both compilers; it also verifies required state dumps and
complete cache/state equality with a full forward. Run it separately with
`pixi run -e fortran-precision native-cpu-terminal-prefix-test`.
The real64-layer/no-dump controls are the **pre-correction provenance above**;
they were not rerun for the logging-only change.

Full real-model short65,2K2048/65 and sustained29/1536 CPU controls remain
**NOT_RUN**; the short17/reset cost already approaches this phase's CPU window.
GPU offload, optimization and speed acceptance are outside this completed
first-stage CPU scope. Historical C/Metal failures and bounds remain unchanged.

### Explicit external LFortran CPU binding (prepared; no real run yet)

The same `run_native_model.py` supervisor also accepts
`--compiler lfortran --lfortran-handoff /absolute/handoff.json --lease /absolute/lease.json`.
It does not build/install anything or introduce another supervisor. Its app
assets and receipts are rooted at the live FN runner's source location, not
the process cwd. For LFortran, cwd and active Pixi must instead be the explicitly
bound compiler worktree **G**:

```sh
cd /Users/ondrej/repos/lfortran-7/.fix-issue/qwen38-compiler/worktrees/metal-precision
pixi run -e llvm11 python /Users/ondrej/repos/qwen3.6/artifacts/worktrees/fortran-native/src-fortran/tests/run_native_model.py \
    --compiler lfortran \
    --lfortran-handoff /Users/ondrej/repos/lfortran-7/.fix-issue/qwen38-compiler/worktrees/metal-precision/.fix-issue/qwen38-native-cpu-final/handoff.json \
    --preflight-only
# Only with a fresh parent grant: replace --preflight-only with
# --lease /Users/ondrej/repos/qwen3.6/artifacts/worktrees/fortran-native/artifacts/fortran-native/<new-lfortran-lease>.json
```

The final handoff explicitly provides `compiler_worktree`, `compiler_path`,
compiler SHA/commit, `pixi_environment=llvm11`, `source_commit=c077fe0`,
`source_archive_root`, build-receipt path/SHA, `binary_root`, binary SHA map
and flags. The supervisor verifies actual files, current G compiler commit,
every mapped archived file against its committed Git blob, and all archived
Fortran files against current FN Fortran identity. New harness/docs commits
do not falsely invalidate the unchanged Fortran source archive.

The final receipt's exact `source_archive_root`/`source_sha256` mapping, successful
compile/link commands and flags, runtime dependency hash and prior actually
loaded-runtime receipt are checked independently. Active `CONDA_PREFIX`,
`PIXI_PROJECT_ROOT`, `PIXI_ENVIRONMENT_NAME` and G cwd must agree; FN/default
activation and DYLD overrides are rejected. An absent/stale mapping, source,
binary, compiler, runtime or receipt fails closed. The old preparation handoff
is a format example, not this final binding or a model grant.

Final identity-only preflight passes for compiler
`d94db958772e31b95e9f12eca8f1c9c65c0dbe88`, compiler SHA
`41f82ed23995f22f6051684cc37acbfe5ae7ee0edb7792ff09684a0d54c1f7e2`,
qwen-raw SHA `7c58825d47d188a3ab3063c9b95fdf6eb133621e7859a57d986fd7fb5c2b03d7`,
and runtime SHA `d43160c237a1f0a07b183efbd01b5fb68fd2686ea99fc5d9fb9be5df852c397a`.
No real LFortran model process has run. Mocked successful admission uses the
same one-use lease, schema3/IDs/lookahead/count/reset checks and RSS monitor.

The **new LFortran-only** lease ceiling is 3600 s, with the same 38 GiB RSS,
≥55% reported-free entry admission and 4 GiB artifact budget. This separate
window does not extend the old Flang/GFortran 1800 s ceilings or repair their
historical results. Numerical gates and all old receipts remain unchanged.

---

## Historical scalar-profile documentation (a70b69e; not the current FN contract)

The CPU raw-token driver now connects the validated math and model I/O.
See [raw-ID inference](#raw-id-inference) below for its CLI, numerical schedule,
toy validation and remaining gates.

`dense_kernels.f90` implements model-free dense linear/FFN, attention and state kernels in
standard Fortran 2018. It depends only on the unchanged `precision_formats`
module and Fortran intrinsics, not C kernels, a C computational shim, Python,
OpenMP, SIMD/vendor intrinsics, or device libraries. Independent outputs use
`DO CONCURRENT` with construct-local indices and `BLOCK`-local temporaries;
reductions and token recurrences remain sequential. All mutable state and
cache storage belongs to the caller; there is no module state or cache.

This is **not a performance or real-27B whole-model acceptance claim**.
Pure-Fortran configuration, raw-ID JSON, safetensors loading, prefill, cached
decode and reset/reuse are available; see [model I/O](MODEL_IO.md).
Tokenizer, chat/NFC, text rendering and vision input are not implemented.
MoE, top-k, and expert paths are deliberately not ported.
LFortran driver validation is a separate parent-owned gate; this driver was
not compiled with LFortran. The parent's model-free kernel results are recorded
below. IEEE lifecycle/flag diagnostics
remain separate, deferred to
[lfortran/lfortran#16342](https://github.com/lfortran/lfortran/issues/16342).

## API and storage

Call `require_precision_environment()` from `precision_formats` at application
startup; the math module has no lazy initialization.

BF16 arrays contain **bit patterns** in `integer(int16)`, not numeric integers.
Packed q8 uses `integer(int32)` bit patterns, low byte first, four unsigned
bytes per word, with affine groups of 64. Arithmetic is `real(real32)` with
explicit BF16 round-to-nearest-even stores via the existing helpers.
`integer(int64)` is used only for checked sizes/index products.

All arrays below are contiguous. Column-major dimensions map directly onto the
existing C flat storage, **without transposing or repacking**:

| API (argument order) | Shapes and behavior |
| --- | --- |
| `embed_dequant_gather_q8(ids, w, scales, offsets, y)` | `ids(M)`, `w(D/4,V)`, `scales/offsets(D/64,V)`, `y(D,M)`; IDs are **0-based**, including 0 and V-1. |
| `linear_q8_bf16(x, w, scales, offsets, y [,bias])` | `x(K,M)`, `w(K/4,N)`, `scales/offsets(K/64,N)`, `y(N,M)`, optional BF16 `bias(N)`; affine offsets are not the optional linear bias. |
| `rmsnorm_bf16(x, y, eps [,weight])` | `x(D,M)`, `y(D,M)`, optional BF16 `weight(D)`; omitted means unweighted. |
| `rmsnorm_scale_bf16(x, y, eps, scalar)` | `x(D,M)`, `y(D,M)`, FP32 `eps/scalar`; scalar is rounded to BF16 before multiplying normalized BF16 values. |
| `silu_mul_bf16(gate, up, out)` | Equal-length BF16 vectors; native BF16 sigmoid boundaries, BF16 SiLU store, then BF16 product (FFN). |
| `silu_mul_f32_gate_bf16(gate, up, out)` | Equal-length BF16 vectors; FP32 sigmoid/SiLU/product, only final BF16 store (RMSNormGated). Not interchangeable with FFN SiLU. |
| `elemwise_sigmoid_mul_bf16(gate, x, out)` | Equal-length BF16 vectors; BF16 sigmoid times x. |
| `residual_add_bf16(x, delta)` | Equal-length BF16 vectors; x is `INTENT(INOUT)`, delta is read-only. |
| `silu_inplace_bf16(x)`, `sigmoid_inplace_bf16(x)` | Single `INTENT(INOUT)` BF16 vector, with native BF16 sigmoid boundaries. |
| `argmax_bf16(logits)` | Nonempty BF16 vector; returns `integer(int32)` **0-based** first maximum, including ties. |
| `compute_g_f32(a, a_log, dt_bias, g)` | BF16 `a(Hv,L)`, `a_log/dt_bias(Hv)`; FP32 `g(Hv,L)`, never narrowed to BF16. |
| `conv1d_depthwise_causal_bf16(x, state, w, y)` | BF16 `x/y(C,L)`, `w(K,C)`, `state(C,K-1)` **in/out**, oldest carry first; K=1 has an empty carry. |
| `gated_delta_step_bf16(q, k, v, g, beta, state, y)` | BF16 `q/k(Dk,Hk)`, `v/y(Dv,Hv)`, `beta(Hv)`; FP32 `g(Hv)` and **in/out** `state(Dk,Dv,Hv)`. One token per call; Hv must be divisible by Hk. |
| `rope_partial_inplace_bf16(x, inv_freqs, d_rot, q_offset)` | BF16 **in/out** `x(D,H,L)`, FP32 `inv_freqs(d_rot/2)`; even `d_rot` in 0..D; split halves, untouched tail. |
| `sdpa_gqa_bf16(q, k, v, out, q_offset, scale, causal)` | BF16 `q/out(D,Nq,Lq)`, `k/v(D,Nkv,Lk)`; FP32 scale, logical causal flag; Nq must be divisible by Nkv. |

Only x in residual/SiLU/sigmoid/RoPE and state in convolution/gated-delta are
`INTENT(INOUT)`; other outputs are `INTENT(OUT)`. **No mutable argument may
overlap any other argument**, including another output. Read-only inputs may
share storage (tested with recurrent q=k and shared decay parameters).
Use distinct caller-owned output/workspace arrays for RMSNorm, linear, and
gate products; never emulate C's input/output aliasing by passing the same
array to different `INTENT(IN)`/`INTENT(OUT)` dummies. Elementwise APIs take
rank-one vectors; a future driver may use contiguous rank-remapped views of
its own storage rather than physically transposing matrices.

Supported shapes are nonempty with each array's element count fitting the
default integer index kind; counts and tile products are checked in int64.
The only empty arrays allowed are convolution's K=1 carry and RoPE's
zero-rotation frequency vector.
Q8 K/D must be positive multiples of 64. Vocabulary bounds, exact shapes,
optional bias/weight lengths, head divisibility, rotation widths, positions
and positive epsilon are checked with explicit `ERROR STOP` diagnostics.
Callers supply finite inference inputs/scales, finite positive epsilon and
the supported storage environment. No exceptional-value policy is added here.

`q_offset` is an **integer(int64), 0-based absolute position**, representing
the C uint32 domain 0..4294967295. The last query position must also fit;
wrapping positions are rejected rather than relying on unsigned overflow.
SDPA's Lk is the **visible cache length**, inferred from the K/V shapes.
Pass contiguous `cache(:,:,1:Lk)` prefixes, not spare cache capacity.
Causal key index t (0-based) is visible iff t <= q_offset + query_index.
Offsets need not be less than Lk; noncausal attention sees every visible key.

Reset state explicitly in caller storage when starting a sequence.
Convolution reads its complete old carry before updating each channel;
short chunks shift the carry left before appending inputs. Gated-delta
finishes each decayed row's k reduction before any rank-one update, then
reduces the updated row against q. Each concurrent iteration owns a distinct
channel or state row. Call gated-delta serially across tokens, retaining the
same FP32 state between calls; no BF16 state checkpoint is permitted.

## Numerical contract

The immutable ground truth is `src-cpu/kernels.{c,h}` at
`97534c69fb8c77e666c1b17f88b6c6048eec73e0`, mathematically unchanged from
`37ff651`:

```
SHA256 kernels.c  694695f941bde8197f8efc82ff5cf40089dab4975c8973c9a663f55c02c5857e
SHA256 kernels.h  4ee4215889b773293f9343470e51b4404ccb236f1b839c2ccb34a5c4a096dc4f
```

In particular:

- RMS squares use compensated FP32 accumulation. Weighted RMSNorm first
  rounds the normalized value to BF16, then multiplies the learned weight.
- BF16 sigmoid uses rounded positive exp, sum, quotient, and subtraction.
  The FP32-gate SiLU variant uses FP32 `g * (1/(1+exp(-g)))` instead.
- Linear preserves the frozen native MLX 0.31.2 M4 shape-dependent precision
  profile. The matrix threshold is M>=18 for K,N<=2048, M>=12 for
  K,N<=4096, otherwise M>=10. These shapes round dequantized weights to
  BF16 and use the C profile's split-K partial BF16 stores and strided
  BF16 reduction (small eight-lane fold versus large 32-lane/tree fold).
  Tile counts determine only this recorded **numerical profile**, not
  CPU/GPU scheduling or an architecture-specific optimization.
- Literal kinds, operation order, optional bias placement, scalar rounding,
  and first-index argmax match C. Compile with FP reassociation/contraction
  disabled; do not silently substitute a different dot product or norm.
- `compute_g` rounds a+dt, exp(-abs(sum)), log1p and softplus to BF16 at
  C's boundaries; exp(A_log) and the final decay remain FP32. Since standard
  Fortran has no LOG1P intrinsic, the implementation corrects rounding in
  `u=1+e` using `log(u)-((u-1)-e)/u`, before the BF16 log1p store.
  The oracle sweep covers every finite BF16 sum image, plus nonzero biases.
- Convolution accumulates taps oldest-to-newest in FP32, with a BF16 output
  store. Recurrent decay, memory reduction, delta, update and output reduction
  retain C's sequential FP32 ordering; only y is stored as BF16.
- RoPE uses FP32 positions/phases/trigonometry and reads both old split-half
  values before either BF16 store. It is not the interleaved-pair variant.
- SDPA preserves C's numerical profiles, not a fused implementation:
  vector shapes have Lq<=8, Lq*(Nq/Nkv)<=32, D in {64,96,128,256};
  full shapes have Lq>8, D in {64,80,128}. Both use FP32 scaled dot scores,
  softmax and V reduction, applying inverse sum **after** the V reduction.
  All other shapes use the declared fallback: BF16 scale and scaled Q,
  BF16 scores and normalized probabilities, then an FP32 V reduction.
  In particular D=256 switches to fallback above eight queries.
  Chunking across a profile boundary is not promised to be bit-identical.

## Model-free checks

Use the existing no-default-feature Pixi environment and approved PATH
compilers. No model, MLX, GPU, global install, or new environment is needed.

```sh
mkdir -p artifacts/fortran-kernels
pixi run -e fortran-precision precision-versions > artifacts/fortran-kernels/versions.log 2>&1
pixi run -e fortran-precision dense-test > artifacts/fortran-kernels/dense-test.log 2>&1
pixi run -e fortran-precision precision-test > artifacts/fortran-kernels/precision-test.log 2>&1
```

The attention/state phase also declares `attention-state-test`, an alias for
the **same complete extended dense harness**, not a second framework:

```sh
mkdir -p artifacts/fortran-attention-state
pixi run -e fortran-precision attention-state-test > artifacts/fortran-attention-state/attention-state.log 2>&1
```

Inspect complete logs. These tasks verify the pinned C/H SHA256 hashes before
building a separate C-only oracle with
`xcrun clang` plus libc/libm, then the actual module with GFortran and Flang in
independent `build/fortran-dense/{gfortran,flang}` module/output directories.
At most two Fortran compiler/test jobs are eligible to run. The new source
and Fortran tests use warnings-as-errors and `-O2 -fno-fast-math
-ffp-contract=off`; GFortran additionally uses `-fcheck=all`. Only the
**unchanged C source's** existing unused `q8_deq` warning is demoted from an
error, not hidden. The known GFortran toolchain duplicate-`-rpath` linker
warning is also retained, not globally suppressed.

The C writer calls the real pinned kernels to generate both inputs and
expected outputs/state in a small explicit native binary fixture (8,121,714 bytes,
ignored under `build/`). The Fortran reader checks integer/float/BF16
representation markers, dimensions, record counts, completeness, and EOF.
The format is test-only/native, not a model loader or portable interchange
format. No C objects are linked into either Fortran executable.

Both GFortran 16.1 and Flang 24 pass **1,366,782 bit-exact output/state comparisons
across 86 records**, with no floating-point tolerance, plus 43 controlled
rejections per compiler. Cases cover M=1 and M>1, all three matrix
thresholds, small/large split-K folds (including 512 partials), divisor search,
512 tiles without splitting, optional bias, signed packed words, group/byte
boundaries, repeated/first/last embedding IDs, RMS D=1 through D=5120 and zero
rows, scale and weighted roundpoints, distinct SiLU variants, sigmoid
saturation/signed zero, residual cancellation, and argmax ties/last index.
The corpus also explicitly witnesses 60 compensated-versus-naive RMS output
differences, 22,403 normalize-before-weight differences, and 5,059
rounded-scalar differences; the C writer fails if these witnesses disappear.
This is exact agreement on this corpus, not a universal cross-libm bitwise
guarantee or real-model acceptance.

The original **897,920** comparisons/51 dense records are retained unchanged.
The added 35 records cover:

- Decay: **65,429 exact FP32** outputs, including every finite BF16 input
  image, signed zero, nonzero a+dt roundpoints and shared read-only parameters.
  34,271 outputs differ from their BF16 narrowing.
- Convolution: K=1,2,4,7 and C=1,3,5,7; six consecutive calls per shape with
  lengths 1,2,1,K+1,2,1. Every output and carry is compared after every call;
  nonzero initial carry and an explicit reset are included.
- RoPE: D=5,7,8,9,256, zero/full/partial rotations, multiple heads, unchanged
  tails, full versus chunked in-place calls, and offsets crossing FP32's exact
  integer range and ending at uint32's maximum.
- SDPA: causal/noncausal, GQA ratios 1,3,4,5,33, D=7,64,80,96,128,256;
  vector/full/fallback paths, Lq=1,8,9, product threshold 32/33/40,
  nonzero offsets and successively visible cache prefixes with poisoned spare
  capacity. The corpus witnesses 69,247 causal-mask output differences.
- Gated-delta: **305,322 exact FP32 state** values plus BF16 outputs checked
  after each of six steps per shape. Includes Dk/Dv=128/128, rectangular/tail
  dimensions, Hv/Hk=48/16, gains 0/1/intermediate FP32, beta 0/1, nonzero
  initial state, reset, and legal read-only q=k aliasing.

Every FP32 state/decay comparison has **zero absolute and ULP error**; no
tolerance was introduced. The 28 added rejection cases cover empty inputs,
shape/head-ratio errors, rotation widths/frequencies and invalid/overflowing
positions. Illegal input/output aliasing is a caller contract violation,
not exercised as if it were a supported runtime operation.
Saved phase logs are under `artifacts/fortran-attention-state/`; each compiler's
individual positive/rejection logs remain under `build/fortran-dense/*/checks/`.
The unchanged precision core was rerun on both compilers: 65,536 BF16 images,
22,032 C FP32 conversions, 256 q8 byte combinations and nine rejections each.

## Raw-ID inference

`dense_model.f90`, `inference_dump.f90` and `main.f90` are standard Fortran
2018. The executable contains no C model/parser/kernel objects, Python,
tokenizer, shell commands or device calls. Test-only Python uses the standard
library to write small HF safetensors and compare with a **separate** unchanged
C executable from `src-cpu` at `97534c6` (math unchanged from `37ff651`).
The oracle build rejects tracked C changes and records source SHA256 hashes.

From this worktree root:

```sh
mkdir -p build/fortran-driver
pixi run -e fortran-precision driver-test > build/fortran-driver/driver-test.log 2>&1
# The small, generated toy model remains available after the test:
pixi run -e fortran-precision driver-gfortran \
  --model build/fortran-driver/checks/model-256 \
  --input-ids build/fortran-driver/checks/matrix-fallback-long/input.json \
  --teacher-ids build/fortran-driver/checks/matrix-fallback-long/teacher.json \
  --max-tokens 17 --fixed-steps --reset-check
```

`driver-build` builds both approved PATH compilers in separate
`build/fortran-driver/{gfortran,flang}` directories. The executables are
`qwen-raw`; `driver-gfortran` and `driver-flang` invoke them without rebuilding.
Build/test tasks serialize the C oracle ahead of the at-most-two Fortran jobs.
Both compile with `-O2 -fno-fast-math -ffp-contract=off`, warnings-as-errors,
and GFortran runtime checks. Only the toolchain's existing duplicate-rpath
linker warnings remain; positive driver runs have no runtime warnings.
No new dependency or global compiler installation is needed.

CLI:

- Required: `--model DIRECTORY --input-ids JSON`. IDs are zero-based integers;
  JSON may be an array or an object containing `input_ids`.
- `--max-tokens N` is the number of **emitted** greedy predictions (default 8,
  positive). Emit first-index argmax after the complete prefill, then perform
  only the one-token forwards needed for the remaining emitted predictions.
  EOS is emitted and stops normal mode; `--fixed-steps` ignores EOS.
- `--teacher-ids JSON` requires `--fixed-steps` and at least N-1 IDs. It
  substitutes the consumed decode inputs, not the reported greedy predictions.
- `--reset-check` clears **all** allocated conv, FP32 delta and KV storage,
  reruns prefill with the same weights, and checks all logits and the complete
  allocated state byte-exactly, including invisible KV capacity.
- `--result FILE` creates a **new** schema-2 JSON report. Otherwise the report
  goes to stdout. Fields include input/generated/consumed IDs, capacities,
  processed-token and forward-call counts, and reset status. Reset forwards
  are reported separately; there is no extra unemitted decode forward.
  Timing/throughput claims are deliberately absent.
- `--check-config` reads only config; `--inspect` additionally validates all
  archive/text metadata and reports counts/bytes, **without payload reads**.
- `--dump DIRECTORY` writes C-compatible little-endian `LLMTNSR1` files:
  BF16 values widened exactly to F32, FP32 gates/state unchanged. The caller
  must create the root, `decode-001` through the needed decode directories,
  and `reset` when requested. Files must not already exist. Fortran does not
  spawn a shell to create directories. Dumps include each C schedule boundary,
  every prefill logit row and each call's visible caches.

The architecture is dense hybrid only. Model dimensions and schedule come
from checked config, not hardcoded target constants. Vision tensors may be
present in the archive but are never selected; vision/text-tokenization CLI
requests and unsupported MoE/config/layout features fail explicitly.
Only already-sanitized weights are supported: no extra RMS +1 or conv
transpose. Q8 affine offsets are not linear biases.

### Runtime API and schedule

`initialize_model(directory,prompt_capacity,context_capacity,model)` owns the
weights and per-layer state in `type(inference_model)`. Capacities must fit
the configured maximum position and every kernel indexing product.
`forward_ids(model,ids,logits,dump_directory)` accepts contiguous IDs, uses
the model's current absolute position, returns all BF16 logit columns and
advances the position. It supports initial multi-token prefill and subsequent
single-token decode; checked per-call workspaces are released on return.
`reset_model(model)` preserves weights and clears state/position. Call
`snapshot_state`/`same_state` only with an initialized model and its matching
snapshot; model configuration/capacity/state shapes must not be modified.
Failures terminate with diagnostics, never success with fallback values.

`same_state` compares FP32 storage with scalar `TRANSFER(real32,int32)` at
every state index, without array-sized bitcast temporaries. This is equivalent
to comparing every word of the original array transfer: signed zeros differ,
and identical quiet-NaN payloads compare equal without real arithmetic.
Both reference compilers cover 43 focused identity/difference checks, including
each element of multiple recurrent layers, zero signs, quiet-NaN payloads,
conv bits and invisible KV capacity. The parent's LFortran explicit-SIZE array
TRANSFER codegen issue remains pending; its original MRE/logs are retained
unchanged in the caller's `.fix-issue/qwen38-driver-probe/`. This portable
comparator change is not a compiler fix or an LFortran driver acceptance claim.

The schedule mirrors C, including:

- Four recurrent projections, causal conv carry then BF16 SiLU, q/k
  RMS-normalize/scale, FP32 `compute_g`, BF16 sigmoid beta, sequential token
  updates of FP32 delta state, weighted RMSNormGated with **FP32** gate SiLU,
  BF16 output projection and a separate BF16 residual.
- Per-head interleaved `(Q,gate)` splitting (not global halves), q/k weighted
  norms, partial split-half RoPE with absolute position and untouched tail,
  time-major-equivalent `(head_dim,kv_heads,capacity)` caches, GQA SDPA and
  BF16 sigmoid gate/output projection, then a separate residual.
- Separate post norm, BF16 FFN SiLU/product, down projection and residual.
  Final norm and head process **all** M rows, retaining C's M-dependent math.

All numerical outputs are disjoint from inputs except the kernels' explicit
`INTENT(INOUT)` forms. Shape-preserving copies/RESHAPE are intentional here;
there is no pointer/rank aliasing or backend-specific optimization.
Existing `DO CONCURRENT` kernel bodies remain unchanged.

### Toy checkpoint (not real-model acceptance)

Both GFortran 16.1 and Flang 24 pass **37,997,136 bit-exact comparisons across
8,280 tensors each**, with zero absolute/ULP tolerance:

- Four-layer H64/FFN128/vocab128 fixtures, three recurrent and one full layer,
  recurrent K/V heads 1/3 and state dimensions 128/128, conv4; full Q/KV
  heads 3/1 with head64 or head256. Q/value widths exceed H.
- Nonzero deterministic q8 weights, affine groups/offsets, norms, gates and
  convolution. Prefix lengths **19, 67, 3, 1**, 17 emitted predictions and
  16 cached decode calls each, both teacher-forced and greedy. This exercises
  matrix/vector q8, full/vector/fallback SDPA and the >64-token prefix.
- **31,232 logits**, **72 call positions**, 68 generated and 64 consumed IDs;
  **11,896,832 cache elements / 576 cache tensors**, comprising 10,616,832
  exact FP32 delta values, 414,720 conv BF16 values and 865,280 KV BF16 values.
- Four reset/reuse comparisons per compiler: **3,523,752 operation elements /
  460 tensors**, including all prefill operations, not just final logits.
  All reset dump files are byte-identical to their original prefill files.
- EOS/fixed-step behavior, object-form IDs, metadata-only inspection,
  fixed-length directory/dump arguments and
  **19 controlled CLI/API rejections** per compiler: bad IDs, capacity,
  activation/index/int64 products, unsupported text/vision/MoE, incompatible
  config and teacher policy/count.

The test asserts a summed <=64 MiB peak resident bound for Python and inference
children (observed below 34 MiB / 17 MiB respectively). Models contain less
than 4 MiB payload each. Successful tests remove the sizable per-operation
dumps; failures retain them to localize the first operation/input mismatch.
Small fixtures, schema-2 reports, rejection logs, `summary.json` and
`resident.json` remain under `build/fortran-driver/checks/`; compiler versions
and `source-pins.txt` are one level above.

Both driver binaries also ran `--inspect` on the real artifact: **six shards,
2,180 tensors / 1,847 text tensors / 28,579,478,528 text bytes**, H5120/FFN17408,
48 recurrent + 16 full layers, Q/value width6144, full Q/KV24/4 head256,
recurrent K/V16/48 head128, conv4, rotary64 and max position262144.
This read headers/config/index only, **zero tensor payload bytes**.
Logs: `build/fortran-driver/actual-metadata-{gfortran,flang}.log`.

Real-27B payload/inference validation still requires an explicit resident-model
lease; no real-model forward or GPU/MLX was run for this checkpoint.
Tokenizer/chat/NFC/rendering, LFortran driver validation and GPU offload remain
separate gates. The parent's model-free LFortran kernel checks pass normal
and f23; its strict optimized reference profile is
`--fast --no-fast-math --skip-pass=fma`, not bare `--fast`.
The separate read-only-mode OPEN defect and optional IEEE exception diagnostics
are not worked around here; model input permissions are never changed.
