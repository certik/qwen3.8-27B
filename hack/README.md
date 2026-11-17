# HACK: src-fortran at the speed of src-cuda (do not merge)

This branch reaches the decode speed of `src-cuda` with the Fortran implementation, by editing the
Fortran and the CUDA that LFortran generates by hand. It is the reference for turning each hack into a
real compiler optimization or a clean Fortran change.

Decode on one H200 NVL, Fibonacci prompt (`--no-think -n 128`), cumulative:

| # | Change | tok/s |
|---|---|---|
| 0 | LFortran `main` (a5126efe16) + this repository's `main` | 113.6 |
| 1 | RMSNorm sum of squares: each producer block adds its sum atomically, in 64-bit fixed point (2^24), into one of two alternating slots (`ssp(1:4)` as two `uint64`); the consumer reads one value | 115.3 |
| 2 | GEMV kernel bodies replaced by src-cuda's `gemv_kernel` (16-byte streaming loads, 64 registers and unroll 2 for the large matrices, 96 registers and unroll 4 for the 1280-block ones) | 117.0 |
| 3 | Gated DeltaNet: src-cuda's `gdn_kernel` (one block per value head, adapted to 256 threads: 64 state values per thread in registers, q/k/v in shared memory, conv state double buffered by position parity) instead of 3 kernels | 118.5 |
| 4 | Attention: src-cuda's `attention_kernel` (cluster of 8 blocks per kv head, online softmax, merged through distributed shared memory; the per-warp merge in two halves to fit 48 KB of static shared memory; KV cache `(d, kh, t)` for both k and v) instead of 3 kernels. Equal to src-cuda launching its kernels one by one (119.3) | 119.3 |
| 5 | Host: the argmax finished on the GPU (`argmax_final`), the token passed to the next step in device memory, the decode steps queued without the host reading anything (the step is inlined into the main program: LFortran waits for the GPU at the return of every procedure that launched kernels) | (enables 6, 7) |
| 6 | Programmatic dependent launch in every kernel: trigger at the start, loads of what the previous kernel does not write (weights, conv state, KV cache) before `griddepcontrol.wait` | ~119.7 |
| 7 | Each token's launches captured into a CUDA graph and replayed (LFortran branch `hack-120` of certik/lfortran, `LF_GRAPH=30`) | **121.1** |

src-cuda: 120.3 tok/s in the same session. Up to step 5 the greedy token ids are identical to src-cuda;
with step 6 the rounding changes and the text differs after token 37 (still a correct function).

## Files

- `../src-fortran/kernels.f90`, `../src-fortran/qwen.f90`: the Fortran of steps 1, 3, 4, 5 (`ssp` slots
  and `k`, placeholder kernels `gdn_all` and `attn_all` whose generated bodies are replaced, `argmax_final`,
  the inlined decode loop).
- `nvccwrap`: an `nvcc` that first applies the script `$LF_PATCH` to every `.cu` input; LFortran uses it
  with `--device-compiler`.
- The scripts that edit the generated CUDA, each applying the previous one:
  `h1.py` (step 1) <- `t1.py` (step 2, `T1PDL=1`: CUDA's PDL prologue) <- `g1.py` / `g1p.py` / `g1p_cg.py`
  (step 3; `g1p*`: PDL, `_cg`: conv state read with `__ldcg`) <- `a1.py` / `a1q.py` / `a1q_cg.py` (step 4;
  `a1q*`: `P1G`, `P1A`, `P1S` select PDL in the GDN, attention and other kernels). The kernel bodies are in
  the `*.cu` files.

## Build and run (step 7)

```console
$ export PATH=<certik/lfortran hack-120 build>/src/bin:$PATH LFORTRAN_LINKER=gcc
$ export LF_PATCH=$PWD/hack/a1q_cg.py T1PDL=1 P1G=1 P1A=1 P1S=1
$ mkdir -p build/hack && cd build/hack && lfortran --fast --realloc-lhs-arrays --gpu=cuda --gpu-overlap \
    --device-compiler $PWD/../../hack/nvccwrap \
    ../../src-fortran/{model_io_support,json_tokens,safetensors_io,qwen_config,qwen_tokenizer,kernels,qwen}.f90 \
    -o qwen && rm -f *.mod
$ LF_GRAPH=30 ./qwen -m "$MODEL_DIR" --no-think -n 128 \
    -p "Write a Fortran function that returns the n-th Fibonacci number."
```

The decode loop runs 127 steps; the reported decode rate counts them all.
