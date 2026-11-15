# Experiments: what would it take to get the GEMVs from ~81% to ~100% of bandwidth?

In `../qwen` (~120 tok/s) the q8 matrix-vector products (GEMVs) are 96% of the time per token and run at
~81% of the measured 4.14 TB/s. This directory records the follow-up investigation (2026-11-15) into
closing that gap. Nothing here is used by `../qwen`.

All numbers: one H200 NVL (mercury GPU 0), best of several runs, one token's 257 GEMVs with the real fused
shapes (27.2 GB of weights + scales + biases). At 4.14 TB/s that is **6.57 ms**.

## 1. Where the 19% goes: a fixed cost per kernel

Each GEMV shape on its own, cycling through enough copies that L2 cannot help (≥1.5 GB):

| Shape (N × K) | Time | Bandwidth time at 4.14 TB/s | Excess |
|---|---|---|---|
| gate_up 34816 × 5120 | 50.1 µs | 45.6 µs | 4.5 µs |
| in_proj 16480 × 5120 | 25.4 µs | 21.6 µs | 3.8 µs |
| q/k/v 14336 × 5120 | 23.3 µs | 18.8 µs | 4.5 µs |
| down 5120 × 17408 | 26.8 µs | 22.9 µs | 3.9 µs |
| out_proj 5120 × 6144 | 11.4 µs | 8.0 µs | 3.4 µs |

The excess is about the same **~4 µs whatever the size**. It is a fixed cost per kernel: launch, the first
DRAM round trip, and the tail as blocks finish unevenly (in a `down` GEMV, blocks that all start
together finish over a ~4 µs spread). In the model, with PDL overlap, it is 4–8 µs per GEMV. That is
257 × ~5 µs ≈ 1.3 ms per token, the whole gap. The inner loop is not the problem: on one 8.5 GB matrix the same
kernel streams at 4.18 TB/s.

Ruled out:
- **DRAM channel camping.** Each block reading its row chunks in a rotated order (so that blocks don't
  hit the same column offset at the same time) changed nothing.
- **Load scheduling.** Taking the pointers from a struct instead of `__restrict__` kernel parameters made a
  standalone harness 20% slower (7.79 vs 9.46 ms per token): nvcc then spreads a step's 4 weight loads over
  ~200 instructions instead of issuing them together. Checked in `../qwen`: same speed either way.
  `../qwen` now passes them as `__restrict__` parameters anyway.
- **Memory footprint.** It costs ~5% (51.1 vs 48.7 µs for gate_up once ≥4 GB of copies rotate). That is part of the excess, not the main part, and one big allocation does not help.

## 2. A persistent kernel can stream at 94–98%

[`persistent_gemv.cu`](persistent_gemv.cu) runs all 257 GEMVs as one cooperative kernel:
- **One block per SM.** Each SM owns a slice of the rows of every matrix.
- **Repacked weights.** Tiles of 4 rows × 1 KB, so one TMA bulk copy (`cp.async.bulk` + `mbarrier`) fetches a tile with its scales.
- **A ring per warp.** Each warp streams its tiles through its own shared-memory ring, several tiles ahead.
- **Streaming across matrices.** Weights never depend on the previous matrix, so the rings keep streaming
  across matrix boundaries. Only the math waits for the grid-wide "previous matrix done" counter.

| Variant | ms per token | Share of 4.14 TB/s |
|---|---|---|
| 257 separate kernels in a CUDA graph (same harness) | 7.84 | 84% |
| Persistent, weight streaming and barriers only (no math, no x staging) | 6.75 | **98%** |
| Persistent, no math (with x staging) | 6.98 | 94% |
| **Persistent, complete** | **9.18** | 72% |

So **streaming across the dependency barriers is solved**: the DRAM pipe can be kept ~95% busy for a whole
token. But with the math the persistent kernel is *slower* than separate kernels.

## 3. The blocker: the dequantization math

In the persistent kernel the math (on tiles already in shared memory) costs about 6 ms of SM time per
token with no weight streaming at all. That is too close to the 6.6 ms of streaming to hide, and the two only
partly overlap with 8–16 warps per SM. The separate kernels get away with it because they run 32 warps
per SM, but those have the fixed per-kernel cost.

The math is already minimal for exact fp32: 3 instructions per weight byte (`PRMT` to build the float
2^23+q, `FADD` to subtract 2^23, `FFMA`). In the 16-blocks-per-SM GEMVs that is ~7% of their time (a
load-only gate_up: 47.3 vs 51.0 µs).

The steps to 90–95% therefore have to be done together:

1. **~3× cheaper math with tensor cores.**
   - Weights convert exactly to fp16 with one `PRMT` per 2 bytes (`0x64qq` is fp16 1024+q) and one `HSUB2`.
   - The activations go in as fp16 hi + lo parts, two columns of the MMA's B operand: ~22 bits, about fp32.
     They need a power-of-two scale per matrix to stay inside fp16 range.
   - One `mma.m16n8k16` then covers 256 weight bytes.
   - The weights have to be repacked at load time into the MMA fragment layout. The checkpoint files stay the
     same, but the in-memory layout no longer matches the Fortran `linear_q8_bf16`.
2. **The persistent kernel for the whole token.** RMSNorm, the residuals and SwiGLU (already GEMV
   epilogues), Gated DeltaNet and attention all become stages of the persistent kernel. That keeps one
   barrier per dependency, which costs ~0.1 ms per token in the prototype.

Expected: 90–95% of bandwidth, about **135–145 tok/s** instead of 120. Not done, because each part is a
substantial project and the gain only appears when both work.

## Prototype log (persistent kernel)

| Version | ms per token | Note |
|---|---|---|
| 2112 small blocks, atomic work counter per matrix | 11.4 | ~1M contended atomics per token |
| Same, static item assignment | 10.7 | 2112 arrivals on one counter per matrix |
| 1 block/SM, 1 row per warp, TMA ring | 57.4 | x re-read from L2 for every row |
| 4-row tiles (repacked), x in shared memory | 13.4 | 6.75 ms without the math |
| One shared ring, 1 producer thread, 16 consumer warps | 14.3 | a single thread can't issue TMA copies fast enough |
| Shared ring, 8 producer lanes | deadlock | after this, every spin loop has a watchdog (`trap`) |
| Per-warp rings, conflict-free (permuted) x | 12.8 | |
| Per-stage tables instead of 64-bit divisions per job | 9.8 | |
| x staging loads batched | **9.2** | the current file |
