# Qwen3.8-27B

Inference for the dense Qwen3.8-27B model (MLX 8-bit checkpoint, group size 64),
in three implementations that read the same safetensors files:

| Directory | What |
|---|---|
| [`src-cpu`](src-cpu) | Single-threaded C99 reference: the numerical specification |
| `src-fortran` | Fortran implementation (CPU, and GPU through LFortran) |
| `src-cuda` | C + CUDA implementation, about 120 tok/s decode on one H200 |

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

On one H200 NVL: 102 tok/s decode (9.8 ms per token), prefill of 5 tokens in 0.7 s.
See [`src-fortran/README.md`](src-fortran/README.md).

## CUDA implementation

A self-contained C + CUDA program (no cuBLAS, no Python): about 120 tok/s decode on one H200 NVL,
with its own tokenizer, safetensors reader and sampling. See [`src-cuda/README.md`](src-cuda/README.md).

```console
$ make cuda
$ src-cuda/qwen -p "What is the capital of France? Answer in one sentence." --no-think
```
