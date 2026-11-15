# Pure-Fortran dense model I/O

These standard Fortran 2018 modules implement **input and storage only**.
There is no C/Python parser or C-kernel linkage. Python's standard library
is used only by the independent tests. The separate pure-Fortran
[raw-ID driver](README.md#raw-id-inference) now connects these modules to
the validated numerical kernels. Existing C, Metal, precision, and math
sources are unchanged.

## Interfaces

Every public reader returns `type(io_error)` from `model_io_support` through
its final required argument, `error`. Check `error%failed()` before using any
output; `error%message` is allocated on failure. Reader calls reset this status.
Failed reads may leave partial output, which must be discarded; they never
report fallback zeros or an empty result as success. All files are closed before
returning; archives own metadata, not open units or mapped payloads.

| Module / entry point | Contract |
| --- | --- |
| `json_tokens`: `read_json(path,doc,error)`, `parse_json(text,doc,error)` | Explicit tokens; root is 1, missing lookup is 0. `member(doc,obj,key)` and `element(doc,array,one_based_index)` navigate without copying subtrees. Strings are decoded UTF-8 bytes; comparisons include length, not Fortran blank padding. Strict number grammar, Unicode/escape validation, duplicate keys, trailing input and depth are checked. No Unicode normalization. |
| `qwen_config`: `read_config(path,cfg,error)` | Reads explicit `text_config`, dense `qwen3_5`/`qwen3_5_text` architecture, dimensions, EOS/BOS/pad, per-layer schedule, affine q8/group64, RoPE and RMS epsilon. Derived widths are checked, not inferred from a directory name. MoE, incompatible gates/activation/dtype, bias, tied embeddings, non-default RoPE, conflicting quantization and invalid dimensions are errors. Positive real parameters must fit normal `real32`. |
| `qwen_config`: `read_input_ids(path,ids,error[,vocab])` | Bare array or object with `input_ids`; nonempty flat integer array only. Returns `integer(int32)` IDs, preserving exact values/order/repeats. IDs are **zero-based**. Optional `vocab` is `integer(int64)` in 1..2^31; otherwise IDs must fit nonnegative int32. Fractional/exponent numbers are not integers for this API. |
| `safetensors_io`: `inspect_shard(path,tensors,shard,error)` | Metadata for one file, including exact names, dtype, int64 shape/elements/bytecount and **zero-based absolute** `byte_offset`. No tensor payload allocation/read. |
| `safetensors_io`: `open_archive(directory,archive,error)` | Uses `model.safetensors.index.json`, since standard Fortran has no directory enumeration API. Checks unique names, complete name-to-shard mapping and optional index `metadata.total_size`. Shard names must be local `.safetensors` basenames; no path traversal. `find_tensor(archive,name)` returns a one-based descriptor index, or 0. |
| `safetensors_io`: `read_bf16(tensor,values,error[,first])`, `read_u32(...)` | Caller-owned flat arrays contain int16 BF16 / int32 U32 **bit images**, not numeric unsigned casts. Read exactly `size(values,kind=int64)` elements beginning at optional int64 `first` (default 1). Ranges cannot exceed the tensor. Rechecks file bounds, including the complete tensor's extent. Partial results after error must be discarded. |
| Same typed readers, rank-2 and BF16 rank-3 overloads | Read a whole tensor into contiguous, preferably `TARGET` arrays with **reversed safetensors dimensions**. Shape/rank must match; these overloads have no `first` argument. Pointer remapping preserves linear storage without a bulk transpose. |
| `qwen_weights`: `validate_dense_weights(archive,cfg,layout,error)` | Separate, metadata-only model semantics: validates every expected text tensor and rejects missing/unexpected text tensors, MTP/unsanitized norms and unsanitized convolution layouts. Returns text count/int64 bytes plus explicit sanitized-norm, packed-input-fastest and convolution-kernel-fastest layout flags. Vision metadata is retained by the generic archive but not selected as text weights. |

`json_document%token` is a flat tree: `kind`, `first`, `count`, and `next`
(the index after a complete subtree). Objects contain alternating key/value
tokens; arrays contain values. Token `text` holds decoded strings or the exact
number spelling. `json_integer`, `json_real`, `json_string`, `json_boolean`,
and `expect_kind` provide checked access. `same_text` compares exact lengths.

## Driver layout handoff

The specification is the pinned `src-cpu/main.c` config/weight validation and
projection splitting, and `src-cpu/kernels.c`. The current artifact is
**already sanitized**: do not add 1 to RMS
weights and do not transpose its convolution weights. MTP *configuration*
alone does not imply MTP weights; the target declares one MTP layer but contains
no MTP tensors. This loader rejects unsanitized inputs rather than guessing a
normalization shift. Generic safetensors reading itself never sanitizes weights.

Safetensors matrices are C-order `[N,K/4]` U32 words and `[N,K/64]` BF16
scales/affine biases. Fortran `w(K/4,N)`, `scales(K/64,N)`,
`offsets(K/64,N)` preserve the bytes and match `dense_kernels` directly.
Affine `biases` are offsets, **not** optional linear projection biases.
Logical q8 byte 0 is bits 0..7, byte 3 bits 24..31, matching
the `precision_formats` module's `packed_q8_byte` procedure.

Other layouts the driver respects:

- Convolution `[C,K,1]` is flat `channel*K+tap`; Fortran `(1,K,C)` or a
  flat view preserves this. `[C,1,K]` is not accepted as sanitized.
- Recurrent QKV projection output is concatenated `Q, K, V`, with widths
  `linear_key_dim, linear_key_dim, linear_value_dim`. Split **activations**,
  not the packed weights.
- Full-attention Q projection has `2*query_heads*head_dim` outputs, ordered
  `[head, (query,gate), component]`; split each head's adjacent Q/gate blocks,
  not two global halves. K/V width is `kv_heads*head_dim`.
- A_log/dt_bias remain BF16 bit vectors. No exp/shift or gate preprocessing
  occurs here. RoPE section/interleave metadata is recorded; multimodal
  position generation is not implemented.

For example, after checking every returned status and missing lookup:

```fortran
use, intrinsic :: iso_fortran_env, only: int32
use model_io_support
use qwen_config
use safetensors_io
use qwen_weights
! Declare cfg, archive, layout, error and t with the types above.
integer(int32), allocatable, target :: w(:,:)
call read_config(model_dir//'/config.json', cfg, error)
! Check error before proceeding after every call.
call open_archive(model_dir, archive, error)
call validate_dense_weights(archive, cfg, layout, error)
t = find_tensor(archive, 'language_model.model.layers.0.linear_attn.in_proj_a.weight')
! Check t /= 0 before indexing.
allocate(w(cfg%hidden/4, cfg%linear_value_heads))
call read_u32(archive%tensor(t), w, error)
```

## Limits and validation

Only the target's **BF16 and U32** dtypes are supported. Scalar or rank 1..6
positive-dimensional tensors are accepted; empty tensors and other formats are
explicitly rejected. Checked signed-int64 arithmetic covers dimensions,
products, bytecounts, offsets and positions. The complete payload must be
nonoverlapping and contiguous, without holes or trailing bytes. Shape widths
for the model remain within existing kernels' default-integer limits; the
loader's flat sizes/positions do not narrow at 2 GiB.

JSON/header text is capped at 16 MiB, one million tokens and 64 nesting levels.
Metadata allocates descriptors/tokens only. Payload reads use at most 64 KiB
staging plus caller-owned output, with no full-model allocation. No `CONVERT=`,
host-endian direct integer reads, unsigned out-of-range numeric conversions,
or floating evaluation of payload bits is used. Little-endian bytes are
assembled explicitly, independent of host byte order. Supported platforms must
have byte-preserving 8-bit character/file units and two's-complement 16/32-bit
integers; these conditions are checked. Both validations ran on little-endian
Darwin arm64; big-endian hardware was not available.

From this worktree root, with the approved PATH GFortran 16.1 and Flang 24:

```sh
mkdir -p build/fortran-io
pixi run -e fortran-precision io-test > build/fortran-io/tests.log 2>&1
pixi run -e fortran-precision io-test-model "$MODEL_DIR" > build/fortran-io/model-tests.log 2>&1
```

The no-default-feature environment declares only test Python, not MLX.
The task DAG builds at most two compilers with independent module directories
under `build/fortran-io/{gfortran,flang}`. Test commands, compiler versions and
each positive/rejection output are saved under that tree; inspect the logs.
Generated fixtures are removed on success (retained for diagnosis on failure);
none are committed.

The target-specific model check independently compares all **2,180** descriptors
in **six** shards against Python `json`/`struct`, checks all **1,847** text
tensors / **28,579,478,528 bytes**, and validates the data-derived
64-layer H5120/inter17408, 48 recurrent/16 full, Q/KV 24/4 head256,
recurrent K/V heads16/48 head128, kernel4, partial-RoPE64/base1e7 configuration.
Only final norm (5,120 BF16), A_log (48 BF16) and 16 packed U32 words are read:
**10,400 bytes** per compiler plus the independent oracle's same small reads.
The final norm is itself beyond a 2 GiB file offset.

Focused synthetic groups cover malformed JSON/Unicode/IDs, strict config and
feature checks, header/data truncation, dtype/rank/dimension/product/range
failures, duplicate/mismapped names, index bytecounts, byte-exact signed bit
images, reversed matrix/cube layouts, chunk boundaries, stale payloads and
a **sparse** >2 GiB seek without loading the large padding tensor.

The reviewed fixed-length-character archive directory joins are corrected
and covered by a regression; both reference compilers pass 128 focused checks.
The raw-ID driver additionally passes complete small synthetic hybrid-model
C comparisons (see README). **Pending:** tokenizer/chat/NFC/rendering, real-27B
inference validation under a model lease, LFortran driver validation and GPU
offload. Metadata/selected-byte checks are not whole-model inference acceptance.
# FN native profile

The loader, archive validation, qwen3_5(_text) architecture and precision
containers below are unchanged. The isolated FN branch now uses native-semantic
math and cache-only prefix generation; see [README.md](README.md#current-fn-native-semantic-cpu-profile)
for current tasks, the changed final-row-only `forward_ids` result, rounded
sampler, schema3 counts, bounded diagnostics and full-request reset checks.
Historical scalar-profile driver/oracle descriptions below are not native-ID
acceptance claims.
