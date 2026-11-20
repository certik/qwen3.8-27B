# Qwen3.8-27B

Inference for the dense Qwen3.8-27B model (MLX 8-bit checkpoint, group size 64),
in three implementations that read the same safetensors files:

| Directory | What |
|---|---|
| [`src-cpu`](src-cpu) | Single-threaded C99 reference: the numerical specification |
| `src-fortran` | Fortran implementation of `src-cuda`, run on the GPU through LFortran, about 119 tok/s decode on one H200 |
| `src-cuda` | C + CUDA implementation, about 120 tok/s decode on one H200 |

## Download the weights

The weights are the LM Studio MLX 8-bit checkpoint
[`lmstudio-community/Qwen3.8-27B-MLX-8bit`](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-8bit)
(6 safetensors files, about 28 GB, group size 64; original model
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)). No login is needed:

```console
$ pip install -U huggingface_hub
$ hf download lmstudio-community/Qwen3.8-27B-MLX-8bit --local-dir "$MODEL_DIR"
```

(`huggingface-cli download` with the same arguments works with older `huggingface_hub`.) All three
implementations read this directory directly (`--model "$MODEL_DIR"`, or `-m` for the CUDA and Fortran programs); it must contain
`model-0000*-of-00006.safetensors`, `model.safetensors.index.json`, `config.json`, `tokenizer.json`
and `vocab.json`. Use a local disk: loading takes about 6 s from cache.

## CPU reference

```console
$ make
$ ./qwen35-cpu --model "$MODEL_DIR" --input-ids ids.json --max-tokens 17 --fixed-steps
```

See [`src-cpu/README.md`](src-cpu/README.md).

## Fortran implementation

Plain Fortran (`do concurrent`, no C, no vendor libraries) that LFortran compiles
for NVIDIA GPUs with `--gpu=cuda` or Apple GPUs with `--gpu=metal`. It mirrors the CUDA
implementation: [`kernels.f90`](src-fortran/kernels.f90) has the kernels of
[`src-cuda/kernels.cu`](src-cuda/kernels.cu) as `do concurrent` loops (one loop per CUDA kernel, three
each for the Gated DeltaNet and the attention), and [`qwen.f90`](src-fortran/qwen.f90) is the program
of `src-cuda/main.c`. The safetensors, config and JSON readers are in `safetensors_io.f90`,
`qwen_config.f90`, `json_tokens.f90` and `model_io_support.f90`.

```console
$ LFORTRAN_LINKER=gcc make fortran-gpu      # needs a recent LFortran main
$ ./qwen-fortran-cuda -m "$MODEL_DIR" --no-think \
    -p "What is the capital of France? Answer in one sentence."
What is the capital of France? Answer in one sentence.

The capital of France is Paris.
```

The tokenizer (byte-level BPE, the Qwen pre-tokenizer, NFC normalization and the chat template) is
written in Fortran too ([`qwen_tokenizer.f90`](src-fortran/qwen_tokenizer.f90)), checked against
HuggingFace `tokenizers` with `make fortran-tokenizer-test`. `-p PROMPT` applies the chat template (one
user message) and streams the answer as text; `--raw` tokenizes the prompt as is, `--no-think` turns
thinking off (it is on by default), `-n N` (default 512) bounds the answer, `-c N` sets the context
capacity (default 8192), `-m DIR` is the model directory and `--ids` prints the prompt and output token
ids to stderr. Generation is greedy.

### NVIDIA GPU

Built with LFortran `main` at `2e12da0696`, run on one H200 NVL (the timing lines go to stderr):

````console
$ ./qwen-fortran-cuda -m "$MODEL_DIR" --no-think -n 128 \
    -p "Write a Fortran function that returns the n-th Fibonacci number."
Write a Fortran function that returns the n-th Fibonacci number.

```fortran
function fib(n) result(f)
    implicit none
    integer, intent(in) :: n
    integer :: f
    integer :: a, b, i

    if (n <= 0) then
        f = 0
    else if (n == 1) then
        f = 1
    else
        a = 0
        b = 1
        do i = 2, n
            f = a + b
            a = b
            b = f
        end do
    end if
end function fib
```
tokenizer .2 s, weights + state in 6.5 s, prompt 25 tokens
prefill: 25 tokens in .848 s (29.5 tok/s); decode: 126 tokens in 1.059 s (119.0 tok/s)
````

Decode speed on one H200 NVL: **about 119 tok/s** (5 runs: 118.9 to 119.4 tok/s, median 119.0); for comparison the
CUDA implementation reaches 120.4 to 120.6 tok/s (median 120.5) in the same runs. The greedy token ids are identical to those of `src-cuda`.
The prompt goes through the same single-token forward pass, one token at a time (prefill is not
optimized).
See [`src-fortran/README.md`](src-fortran/README.md).

### Apple GPU (Metal)

On Apple silicon, use a recent LFortran on `PATH` and the Xcode Command Line Tools.
Set `MODEL_DIR` to the downloaded MLX 8-bit checkpoint directory (see above), then
build from the repository root:

```console
$ lfortran --fast --realloc-lhs-arrays --gpu=metal --gpu-overlap \
    src-fortran/model_io_support.f90 src-fortran/json_tokens.f90 \
    src-fortran/safetensors_io.f90 src-fortran/qwen_config.f90 \
    src-fortran/qwen_tokenizer.f90 src-fortran/kernels.f90 \
    src-fortran/qwen.f90 -o qwen-fortran-metal
$ ./qwen-fortran-metal -m "$MODEL_DIR" --no-think \
    -p "Write a Fortran function that returns the n-th Fibonacci number."
```

The recorded Apple M4 Max (64 GiB unified memory) steady-decode baseline is
**3.42 tok/s**, measured over 63 timed tokens in a 64-token fixed-step run;
model loading and prefill are excluded. It was measured with the previous Fortran
implementation (which produced the same Fibonacci function on the Apple GPU as on
the H200) and has not been re-measured with the current one.
The tested source revisions were:

- Qwen Fortran `main`: `4ac8c553536a4cdb444d9c757592376220ff7dba`.
- LFortran `main`: `5d6dcfa371518308306510adf53a88a42de5bc28` (LLVM 11.1.0).

## CUDA implementation

A self-contained C + CUDA program (no cuBLAS, no Python): about 120 tok/s decode on one H200 NVL,
with its own tokenizer, safetensors reader and sampling. See [`src-cuda/README.md`](src-cuda/README.md).

```console
$ make cuda
$ src-cuda/qwen -p "What is the capital of France? Answer in one sentence." --no-think
```
