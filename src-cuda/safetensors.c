#include "safetensors.h"
#include "json.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char* path;
    const uint8_t* map;
    size_t size;
    jval* header;
    size_t data_offset;
} shard;

typedef struct { char* name; int shard; } entry;

struct st_model {
    shard* shards;
    int n_shards;
    entry* entries;   // sorted by name
    int n_entries;
};

static int cmp_entry(const void* a, const void* b) { return strcmp(((const entry*)a)->name, ((const entry*)b)->name); }

static int open_shard(shard* s) {
    int fd = open(s->path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "safetensors: cannot open %s\n", s->path); return 0; }
    struct stat sb;
    fstat(fd, &sb);
    s->size = (size_t)sb.st_size;
    s->map = mmap(NULL, s->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (s->map == MAP_FAILED) { fprintf(stderr, "safetensors: mmap failed for %s\n", s->path); return 0; }
    uint64_t hlen;
    memcpy(&hlen, s->map, 8);
    s->header = json_parse((const char*)s->map + 8, hlen);
    s->data_offset = 8 + hlen;
    return s->header != NULL;
}

static int add_shard(st_model* m, const char* dir, const char* file) {
    for (int i = 0; i < m->n_shards; i++)
        if (!strcmp(strrchr(m->shards[i].path, '/') + 1, file)) return i;
    m->shards = realloc(m->shards, (m->n_shards + 1) * sizeof(shard));
    shard* s = &m->shards[m->n_shards];
    memset(s, 0, sizeof(*s));
    s->path = malloc(strlen(dir) + strlen(file) + 2);
    sprintf(s->path, "%s/%s", dir, file);
    if (!open_shard(s)) return -1;
    return m->n_shards++;
}

st_model* st_open(const char* dir) {
    st_model* m = calloc(1, sizeof(st_model));
    char path[4096];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    FILE* f = fopen(path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        size_t n = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        char* text = malloc(n);
        if (fread(text, 1, n, f) != n) n = 0;
        fclose(f);
        jval* idx = json_parse(text, n);
        free(text);
        const jval* wm = json_get(idx, "weight_map");
        if (!wm) { fprintf(stderr, "safetensors: bad index %s\n", path); json_free(idx); st_close(m); return NULL; }
        m->entries = malloc(wm->len * sizeof(entry));
        for (size_t i = 0; i < wm->len; i++) {
            int si = add_shard(m, dir, wm->items[i].str);
            if (si < 0) { json_free(idx); st_close(m); return NULL; }
            m->entries[m->n_entries].name = strdup(wm->keys[i]);
            m->entries[m->n_entries].shard = si;
            m->n_entries++;
        }
        json_free(idx);
    } else {   // single-file checkpoint
        int si = add_shard(m, dir, "model.safetensors");
        if (si < 0) { st_close(m); return NULL; }
        const jval* h = m->shards[si].header;
        m->entries = malloc(h->len * sizeof(entry));
        for (size_t i = 0; i < h->len; i++) {
            if (!strcmp(h->keys[i], "__metadata__")) continue;
            m->entries[m->n_entries].name = strdup(h->keys[i]);
            m->entries[m->n_entries].shard = si;
            m->n_entries++;
        }
    }
    qsort(m->entries, m->n_entries, sizeof(entry), cmp_entry);
    return m;
}

void st_close(st_model* m) {
    if (!m) return;
    for (int i = 0; i < m->n_shards; i++) {
        if (m->shards[i].map && m->shards[i].map != MAP_FAILED) munmap((void*)m->shards[i].map, m->shards[i].size);
        json_free(m->shards[i].header);
        free(m->shards[i].path);
    }
    for (int i = 0; i < m->n_entries; i++) free(m->entries[i].name);
    free(m->shards);
    free(m->entries);
    free(m);
}

int st_find(const st_model* m, const char* name, st_tensor* out) {
    entry key = {(char*)name, 0};
    const entry* e = bsearch(&key, m->entries, m->n_entries, sizeof(entry), cmp_entry);
    if (!e) return 0;
    const shard* s = &m->shards[e->shard];
    const jval* t = json_get(s->header, name);
    if (!t) return 0;
    const jval* dtype = json_get(t, "dtype");
    const jval* shape = json_get(t, "shape");
    const jval* offs = json_get(t, "data_offsets");
    memset(out, 0, sizeof(*out));
    snprintf(out->dtype, sizeof(out->dtype), "%s", dtype->str);
    out->ndim = (int)shape->len;
    for (int i = 0; i < out->ndim && i < 8; i++) out->shape[i] = (int64_t)shape->items[i].num;
    size_t b = (size_t)offs->items[0].num, e2 = (size_t)offs->items[1].num;
    out->data = s->map + s->data_offset + b;
    out->nbytes = e2 - b;
    return 1;
}
