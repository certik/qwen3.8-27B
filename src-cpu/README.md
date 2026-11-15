# src-cpu — dense/MoE Qwen 3.5 (MLX q8) scalar C reference

A standalone C99 implementation of dense Qwen3.8-27B and original
`Qwen3.6-35B-A3B-MLX-8bit` MoE inference,
intended as a numerical and structural specification of the model — not as a
fast runtime. Built to be the starting point for Apple-GPU Metal ports (and
later for CUDA / ROCm / TPU / NPU ports), all of which can be validated
against the per-op oracle dumps produced from MLX.

## Status

The results below are historical Qwen3.6 observations, not current acceptance
evidence. Synthetic checks cover SDPA, convolution, recurrence, MoE, and
recurrent state. The dense driver selects the FFN from configuration, validates
native q8/BF16 layouts before use, and sizes activation and KV workspaces
separately. Whole-model acceptance is still pending; kernel success alone is
not model acceptance.

Historically, eight-token greedy decode was reported to match:

```
$ ./qwen35-cpu --prompt "Hello, who are you?" --max-tokens 8
prompt: 16 tokens
loading weights from $MODEL_DIR
prefill (Lp=16) ... prefill: ~20 s   first id=8160 ("Here")
Here's a thinking process:

1
```

The 8 generated IDs `[8160, 579, 264, 7047, 1817, 25, 271, 16]` are
byte-identical to MLX's greedy output (verified via `src-mlx/inference.py`
with `temp=0`).

The old per-kernel error report (not a trustworthy pass/fail report) was:

| kernel                                       | max\|err\|        | notes                          |
|----------------------------------------------|---------------|--------------------------------|
| `embed_dequant_gather_q8`                    | 0             | exact                          |
| `rmsnorm_bf16` (input_layernorm L0)          | 0.125         | 1 bf16 ULP at \|x\|≈24         |
| `linear_q8_bf16` (q_proj L3)                 | 0.0625        | 2048-K bf16 accumulation       |
| `silu_mul_bf16` (SwiGLU mid)                 | 0.0078        | ~1 bf16 ULP                    |
| `compute_g_bf16`                             | 0.0024        |                                |
| `softmax_topk_bf16`                          | 0             | 0 set mismatches over Lp×K     |
| `rope_partial_inplace_bf16` (q L3)           | 0             | exact                          |
| `sdpa_gqa + sigmoid_mul + o_proj` (L3 e2e)   | 0.002         | over 32 768 elements           |
| `linear_q8_gather_bf16` (MoE gate/up/down)   | 0.0005–0.002  |                                |
| `expert_mix_bf16`                            | 0.0005        |                                |
| `conv1d_depthwise_causal_bf16 + silu`        | 0.031         |                                |
| `gated_delta_step_bf16` (sequential L=Lp)    | 0.008         |                                |

## Layout

```
src-cpu/
├── main.c           # CLI, forward(), AR decode loop
├── kernels.c        # 16 naive scalar kernels (~600 lines)
├── kernels.h
├── safetensors.{c,h}# mmap'd HF safetensors loader (BF16/F16/F32/U8/I8/U32/I32)
├── tokenizer.{c,h}  # byte-level BPE + Qwen pre-tokenizer
├── tokenizer.bin    # built from tokenizer.json by src-cpu/build_tokenizer.c
├── build_tokenizer.c# tokenizer.json -> tokenizer.bin (uses utils/json + utils/bytebuf)
├── utils/           # reusable, model-agnostic helpers (no Qwen knowledge)
│   ├── json.{h,c}   # Python-like JSON reader on top of vendored jsmn
│   ├── jsmn.h       # vendored JSON tokenizer (MIT, Serge Zaitsev)
│   ├── utf8.{h,c}   # UTF-8 codec (encode/decode one codepoint)
│   ├── bytebuf.{h,c}# growable byte buffer (Python's bytearray)
│   ├── iofile.{h,c} # read-only file mmap helper
│   └── bf16.h       # bf16 <-> f32 round-to-nearest-even
├── qwen_chat.{c,h}  # chat-template builder for Qwen 3.5
├── refs/            # MLX per-op oracle dumps (regenerable via tools/dump_ref.py)
└── tests/
    ├── load_ref.h
    └── test_kernels.c
```

## Architecture summary

The dense artifact declares `qwen3_5/qwen3_5_text`, independently of its
directory name: 64 layers, hidden 5120, FFN 17408, Q/KV heads 24/4 with
head dimension 256, recurrent K/V heads 16/48 with dimension 128. Forty-eight
layers are recurrent and sixteen full attention. Q and recurrent value
widths are 6144, not hidden width. Only native q8 group-64 weights, BF16
activations/scales/biases and FP32 decay/recurrent state are supported.
Text tensors are mmap-accessed on demand; no vision arrays are materialized.

The scalar arithmetic preserves native shape-dependent BF16
dequantization/split-K partials and unfused SDPA score/probability stores,
without introducing optimized CPU kernels.

Original MoE configuration:

- **40 layers**, hybrid:
  - 30 layers of **GatedDeltaNet** (mamba-like recurrence) on `(i+1) % 4 != 0`
  - 10 layers of **full attention** (GQA 16 Q / 2 KV, head_dim=256) on `(i+1) % 4 == 0`
- **MoE every layer**: 256 experts, top-8, with shared expert + shared-expert gate
- All Linears + Embedding are **8-bit affine** (group_size=64, MLX format)
- **Output gate** on full attention (`out *= sigmoid(per_head_gate)`)
- **Partial RoPE** (rotate first 64 of 256 head_dim; base=1e7) on full-attn only
- **RMSNorm** everywhere (eps=1e-6), including per-head `q_norm`/`k_norm`
- **GatedDeltaNet**: depthwise causal Conv1d (kernel=4), in_proj_{qkv,z,a,b},
  recurrent state `[Hv=32, Dv=128, Dk=128]` per layer, RMSNormGated output

## Distinct kernels (16)

1. `embed_dequant_gather_q8` — dequant 8-bit embedding row by token id
2. `rmsnorm_bf16` — RMSNorm with optional weight
3. `rmsnorm_scale_bf16` — RMSNorm with external scalar (GatedDeltaNet q/k_norm)
4. `linear_q8_bf16` — dense Linear with 8-bit weights
5. `linear_q8_gather_bf16` — per-expert gather + dequant + matmul (MoE)
6. `rope_partial_inplace_bf16` — partial rotary embedding
7. `sdpa_gqa_bf16` — GQA scaled dot-product attention with causal mask
8. `elemwise_sigmoid_mul_bf16` — `sigmoid(gate) * x` (output gate, shared gate)
9. `residual_add_bf16`
10. `silu_mul_bf16` — SwiGLU: `silu(gate) * up`
11. `silu_inplace_bf16` — used inside conv1d branch
12. `sigmoid_inplace_bf16` — used for `beta`
13. `softmax_topk_bf16` — router: softmax → top-K → renorm
14. `expert_mix_bf16` — weighted sum over selected experts
15. `conv1d_depthwise_causal_bf16` — depthwise causal Conv1d with carried state
16. `compute_g_f32` — `exp(-exp(A_log) * softplus(a + dt_bias))`, FP32 output
17. `gated_delta_step_bf16` — one recurrent step of the delta-rule SSM
18. `argmax_bf16` — greedy sampling
19. `silu_mul_f32_gate_bf16` — RMSNormGated's FP32 SiLU/product, BF16 store

## Build & run

```
make                # qwen35-cpu, with contraction and vectorization disabled
make tests          # model-free precision checks
```

Pass an explicit `--model PATH` to the binary. Its existing Qwen3.6 default
is historical. Dense inputs require `--input-ids FILE`, a strict nonempty JSON
integer array. Dense text and chat-template input are rejected. Old MoE
`--prompt` remains available.

```
./qwen35-cpu --model "$MODEL_DIR" --input-ids ids.json \
  --max-tokens 17 --fixed-steps --reset-check \
  --dump artifacts/cpu-short --result artifacts/cpu-short.json
```

`--teacher-ids FILE` supplies fixed continuation inputs and requires
`--fixed-steps`. Otherwise greedy outputs feed subsequent calls. EOS is
emitted and stops normal mode; fixed-step mode records it and continues.
Exactly one prefill plus generated-minus-one cached calls execute (no MLX
lookahead). `--reset-check` clears/reuses all allocated caches and reruns the
original prefill; all logits and all caches must be byte-exact. Dump/result
paths must be new, never silently overwritten. Dumps are `LLMTNSR1` widened
F32, including every layer's intermediates and both cache/state slots;
attention cache layout is `[tokens,heads,dim]`.

The result separates startup, effective prefill/TTFT (embedding through
full-position logits and final-position argmax), decode, inference and process
wall including optional reset. Effective prefill rate is prompt tokens/TTFT;
it is not isolated GPU compute. Dump I/O is included when requested, and these
correctness runs are always marked invalid as throughput benchmarks. Future
performance comparisons still require short chat, 2K prefill and sustained
decode under the same documented protocol.
Startup maps headers/weights but does not eagerly read the whole payload;
weight page faults belong to the subsequent measured inference, not to an
invented eager weight-loading measurement.
The reported `process_wall_seconds` spans C `main` entry through its summary
timestamp, including reset, but excludes final result writing. It is not an
externally timed complete-process benchmark.

## Running the per-kernel tests

The tests in `src-cpu/tests/test_kernels.c` compare each C kernel against
binary oracle dumps captured from the MLX reference, kept in
`src-cpu/refs/*.bin`. Those dumps are not checked into the repo because
they're large and trivially regenerable.

Generate them once, then run the tests:

```
python tools/dump_ref.py --model "$MODEL_DIR" --out artifacts/oracle
make src-cpu/tests/test_kernels
./src-cpu/tests/test_kernels --model "$MODEL_DIR" --refs artifacts/oracle
```

`dump_ref.py` writes every per-op intermediate (input_ids, embed output,
RMSNorm outputs, q/k/v projections, RoPE outputs, SDPA output, router
logits, MoE branch outputs, etc.) for layers 0 and 3 (the first linear-
attn and first full-attn layer respectively). The capture schema is at the
top of `tools/dump_ref.py`. `src-cpu/tests/test_precision_replay` replays
captured final RMSNorm, `lm_head`, and selected q8 projections when those
fixtures are available.

Re-run `dump_ref.py` whenever you change the prompt, model, or capture
schema. Never overwrite an oracle to make a C failure disappear.

## Performance

Historical original-MoE observations, not rerun and **not dense-target
measurements**: on one Apple-silicon CPU core (clang `-O2`, no SIMD, no threads), with
all 8-bit weights dequantized inside the matmul inner loop:

- prefill (Lp = 16, all 40 layers): ~20 s
- AR decode (per token):            ~1.5 s

This is intentional. The C reference is a spec, not a runtime. Speed comes
in `port-c-to-metal` (1:1 Metal port) and then `optimize-metal`
(SIMD-group GEMV, qmv tiling, MMA prefill, etc.).
