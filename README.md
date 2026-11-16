# Qwen3.8-27B

Inference for the dense Qwen3.8-27B model (MLX 8-bit checkpoint, group size 64),
in three implementations that read the same safetensors files:

| Directory | What |
|---|---|
| [`src-cpu`](src-cpu) | Single-threaded C99 reference: the numerical specification |
| `src-fortran` | Fortran implementation (CPU, and GPU through LFortran) |
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
implementations read this directory directly (`--model "$MODEL_DIR"`); it must contain
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
for NVIDIA GPUs with `--gpu=cuda` or Apple GPUs with `--gpu=metal`:

```console
$ LFORTRAN_LINKER=gcc make fortran-gpu      # needs a recent LFortran main
$ ./qwen-fortran-cuda --model "$MODEL_DIR" --no-think \
    --prompt "What is the capital of France? Answer in one sentence."
The capital of France is Paris.
```

The tokenizer (byte-level BPE, the Qwen pre-tokenizer, NFC normalization and the chat template) is
written in Fortran too ([`qwen_tokenizer.f90`](src-fortran/qwen_tokenizer.f90)), checked against
HuggingFace `tokenizers` with `make fortran-tokenizer-test`. `--prompt` applies the chat template and
streams the answer as text; `--raw` tokenizes the prompt as is, `--system TEXT` adds a system message,
`--think` (default) / `--no-think` selects thinking mode, `--prompt-file` reads the prompt from a file,
`--tokenize TEXT` / `--detokenize ids.json` run the tokenizer alone (no weights), and
`--max-tokens N` (default 256) bounds the answer. For benchmarking, `--input-ids ids.json --fixed-steps
--result out.json` runs on token ids and writes a JSON report.

### NVIDIA GPU demo

Built with LFortran `main` at `6855c1f679`, run on one H200 NVL (answers are generated on the GPU;
`timing:` lines go to stderr):

````console
$ ./qwen-fortran-cuda --model "$MODEL_DIR" --no-think \
    --prompt "Write a Fortran function that returns the n-th Fibonacci number."
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
timing: load    5.865 s, prefill    2.321 s, first decode step    0.011 s
timing: steady decode 126 tokens     1.240 s =   101.62 tok/s, median step     9.81 ms, max    10.89 ms
````

Decode speed on one H200 NVL: **about 102 tok/s** (9.8 ms per token; 5 runs: 101.99 to 102.36 tok/s).
For comparison the CUDA implementation reaches about 120 tok/s. Prefill is not optimized yet
(about 2.3 s for a 24-token prompt).
See [`src-fortran/README.md`](src-fortran/README.md).

### Apple GPU (Metal)

On Apple silicon, use a recent LFortran on `PATH` and the Xcode Command Line Tools.
Set `MODEL_DIR` to the downloaded MLX 8-bit checkpoint directory (see above), then
build from the repository root:

```console
$ lfortran --fast --gpu=metal --gpu-overlap \
    src-fortran/precision_formats.f90 src-fortran/dense_kernels.f90 \
    src-fortran/model_io_support.f90 src-fortran/json_tokens.f90 \
    src-fortran/safetensors_io.f90 src-fortran/qwen_config.f90 \
    src-fortran/qwen_weights.f90 src-fortran/inference_dump.f90 \
    src-fortran/dense_model.f90 src-fortran/qwen_tokenizer.f90 \
    src-fortran/main.f90 -o qwen-fortran-metal
$ ./qwen-fortran-metal --model "$MODEL_DIR" --no-think \
    --prompt "Write a Fortran function that returns the n-th Fibonacci number."
```

This runs inference on the Apple GPU and produces the same Fibonacci function as
the H200 demo above.

The recorded Apple M4 Max (64 GiB unified memory) steady-decode baseline is
**3.42 tok/s**, measured over 63 timed tokens in a 64-token fixed-step run;
model loading and prefill are excluded. This is the initial, unoptimized baseline.
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
