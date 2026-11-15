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
