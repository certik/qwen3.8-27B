// Read-only access to a sharded safetensors checkpoint (model.safetensors.index.json + shards), via mmap.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const void* data;   // points into the mmap'ed shard
    size_t nbytes;
    char dtype[8];      // "BF16", "U32", "F32", ...
    int ndim;
    int64_t shape[8];
} st_tensor;

typedef struct st_model st_model;

st_model* st_open(const char* model_dir);   // NULL on error
void st_close(st_model* m);
int st_find(const st_model* m, const char* name, st_tensor* out);   // 1 if found
