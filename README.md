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
for the GPU with `--gpu=cuda`:

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

### Demo

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

## CUDA implementation

A self-contained C + CUDA program (no cuBLAS, no Python): about 120 tok/s decode on one H200 NVL,
with its own tokenizer, safetensors reader and sampling. See [`src-cuda/README.md`](src-cuda/README.md).

```console
$ make cuda
$ src-cuda/qwen -p "What is the capital of France? Answer in one sentence." --no-think
```
