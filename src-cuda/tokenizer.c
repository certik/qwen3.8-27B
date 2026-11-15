// Byte-level BPE tokenizer (the HuggingFace "tokenizers" BPE model as configured in
// Qwen's tokenizer.json):
//
//   1. Split the text on added tokens (<|im_start|>, <think>, ...), which map to one id each.
//   2. Pre-tokenize every other span with the Qwen regex (hand-coded below):
//        (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
//        | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//   3. Map each piece's bytes to the GPT-2 "byte-level" characters and apply the BPE merges
//      by rank, then look up the ids.
//
// Not implemented: the NFC normalizer. It does not change ASCII or already-composed text.

#include "tokenizer.h"
#include "json.h"
#include "unicode_tables.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- string hash map

typedef struct { char* key; int32_t val; } hentry;
typedef struct { hentry* e; size_t cap; } hmap;   // open addressing, cap = power of 2

static uint64_t hash_bytes(const char* s, size_t n) {
    uint64_t h = 1469598103934665603ull;   // FNV-1a
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ull; }
    return h;
}

static void hmap_init(hmap* m, size_t n) {
    m->cap = 1;
    while (m->cap < 2 * n) m->cap <<= 1;
    m->e = calloc(m->cap, sizeof(hentry));
}

static void hmap_put(hmap* m, char* key, int32_t val) {   // takes ownership of key
    size_t i = hash_bytes(key, strlen(key)) & (m->cap - 1);
    while (m->e[i].key) {
        if (!strcmp(m->e[i].key, key)) { free(key); m->e[i].val = val; return; }
        i = (i + 1) & (m->cap - 1);
    }
    m->e[i].key = key;
    m->e[i].val = val;
}

static int32_t hmap_get(const hmap* m, const char* key, size_t n) {   // key need not be NUL-terminated
    size_t i = hash_bytes(key, n) & (m->cap - 1);
    while (m->e[i].key) {
        if (!strncmp(m->e[i].key, key, n) && m->e[i].key[n] == 0) return m->e[i].val;
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}

static void hmap_free(hmap* m) {
    for (size_t i = 0; i < m->cap; i++) free(m->e[i].key);
    free(m->e);
}

// ---------------------------------------------------------------- tokenizer

typedef struct { char* text; size_t len; int32_t id; } added_token;

struct tokenizer {
    hmap vocab;            // byte-level token string -> id
    hmap merges;           // "left right" -> rank
    char** id_bytes;       // id -> raw bytes (decoded from the byte-level form)
    size_t* id_len;
    int n_ids;
    added_token* added;
    int n_added;
    char byte_enc[256][3];   // byte -> UTF-8 of its byte-level character (1 or 2 bytes, NUL-terminated)
    int byte_dec[512];       // byte-level code point (< 512) -> byte, or -1
};

static size_t utf8_decode(const char* s, size_t n, uint32_t* cp) {
    const unsigned char* u = (const unsigned char*)s;
    if (u[0] < 0x80 || n < 2) { *cp = u[0]; return 1; }
    if ((u[0] & 0xE0) == 0xC0) { *cp = ((u[0] & 0x1F) << 6) | (u[1] & 0x3F); return 2; }
    if ((u[0] & 0xF0) == 0xE0 && n >= 3) { *cp = ((u[0] & 0x0F) << 12) | ((u[1] & 0x3F) << 6) | (u[2] & 0x3F); return 3; }
    if (n >= 4) { *cp = ((u[0] & 0x07) << 18) | ((u[1] & 0x3F) << 12) | ((u[2] & 0x3F) << 6) | (u[3] & 0x3F); return 4; }
    *cp = u[0];
    return 1;
}

static void init_byte_maps(tokenizer* t) {
    // GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to 256, 257, ...
    int n = 0;
    for (int i = 0; i < 512; i++) t->byte_dec[i] = -1;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t cp = printable ? (uint32_t)b : (uint32_t)(256 + n++);
        t->byte_dec[cp] = b;
        if (cp < 0x80) { t->byte_enc[b][0] = (char)cp; t->byte_enc[b][1] = 0; }
        else { t->byte_enc[b][0] = (char)(0xC0 | (cp >> 6)); t->byte_enc[b][1] = (char)(0x80 | (cp & 0x3F)); t->byte_enc[b][2] = 0; }
    }
}

static char* read_file(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(*n + 1);
    if (fread(buf, 1, *n, f) != *n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[*n] = 0;
    return buf;
}

static void set_id(tokenizer* t, int32_t id, char* bytes, size_t len) {
    if (id < 0 || id >= t->n_ids) { free(bytes); return; }
    free(t->id_bytes[id]);
    t->id_bytes[id] = bytes;
    t->id_len[id] = len;
}

tokenizer* tok_load(const char* path) {
    size_t n;
    char* text = read_file(path, &n);
    if (!text) { fprintf(stderr, "tokenizer: cannot read %s\n", path); return NULL; }
    jval* root = json_parse(text, n);
    free(text);
    if (!root) return NULL;
    const jval* model = json_get(root, "model");
    const jval* vocab = json_get(model, "vocab");
    const jval* merges = json_get(model, "merges");
    const jval* added = json_get(root, "added_tokens");
    if (!vocab || vocab->type != J_OBJ || !merges || merges->type != J_ARR) {
        fprintf(stderr, "tokenizer: unsupported tokenizer.json\n");
        json_free(root);
        return NULL;
    }
    tokenizer* t = calloc(1, sizeof(tokenizer));
    init_byte_maps(t);

    int max_id = -1;
    for (size_t i = 0; i < vocab->len; i++) if ((int)vocab->items[i].num > max_id) max_id = (int)vocab->items[i].num;
    for (size_t i = 0; added && i < added->len; i++) {
        int id = (int)json_get(&added->items[i], "id")->num;
        if (id > max_id) max_id = id;
    }
    t->n_ids = max_id + 1;
    t->id_bytes = calloc(t->n_ids, sizeof(char*));
    t->id_len = calloc(t->n_ids, sizeof(size_t));

    hmap_init(&t->vocab, vocab->len);
    for (size_t i = 0; i < vocab->len; i++) {
        const char* s = vocab->keys[i];
        int32_t id = (int32_t)vocab->items[i].num;
        hmap_put(&t->vocab, strdup(s), id);
        // Raw bytes of the token: every byte-level character maps back to one byte.
        size_t sl = strlen(s), k = 0, m = 0;
        char* raw = malloc(sl + 1);
        while (k < sl) {
            uint32_t cp;
            k += utf8_decode(s + k, sl - k, &cp);
            raw[m++] = (cp < 512 && t->byte_dec[cp] >= 0) ? (char)t->byte_dec[cp] : '?';
        }
        set_id(t, id, raw, m);
    }

    hmap_init(&t->merges, merges->len);
    for (size_t i = 0; i < merges->len; i++) {
        const jval* mg = &merges->items[i];
        char* key;
        if (mg->type == J_STR) key = strdup(mg->str);   // older format: "left right"
        else {
            const char* a = mg->items[0].str;
            const char* b = mg->items[1].str;
            key = malloc(strlen(a) + strlen(b) + 2);
            sprintf(key, "%s %s", a, b);
        }
        hmap_put(&t->merges, key, (int32_t)i);
    }

    t->n_added = added ? (int)added->len : 0;
    t->added = calloc(t->n_added ? t->n_added : 1, sizeof(added_token));
    for (int i = 0; i < t->n_added; i++) {
        const jval* a = &added->items[i];
        const jval* c = json_get(a, "content");
        t->added[i].text = strdup(c->str);
        t->added[i].len = c->len;
        t->added[i].id = (int32_t)json_get(a, "id")->num;
        set_id(t, t->added[i].id, strdup(c->str), c->len);
    }
    json_free(root);
    return t;
}

void tok_free(tokenizer* t) {
    if (!t) return;
    hmap_free(&t->vocab);
    hmap_free(&t->merges);
    for (int i = 0; i < t->n_ids; i++) free(t->id_bytes[i]);
    free(t->id_bytes);
    free(t->id_len);
    for (int i = 0; i < t->n_added; i++) free(t->added[i].text);
    free(t->added);
    free(t);
}

int tok_vocab_size(const tokenizer* t) { return t->n_ids; }

const char* tok_decode(const tokenizer* t, int32_t id, size_t* len) {
    if (id < 0 || id >= t->n_ids || !t->id_bytes[id]) { *len = 0; return ""; }
    *len = t->id_len[id];
    return t->id_bytes[id];
}

// ---------------------------------------------------------------- pre-tokenizer

static int in_ranges(uint32_t cp, const uint32_t (*r)[2], int n) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid - 1;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}
static int is_L(uint32_t c) { return in_ranges(c, UNI_LETTER, sizeof(UNI_LETTER) / sizeof(UNI_LETTER[0])); }
static int is_N(uint32_t c) { return in_ranges(c, UNI_NUMBER, sizeof(UNI_NUMBER) / sizeof(UNI_NUMBER[0])); }
static int is_S(uint32_t c) {   // Unicode White_Space (what \s means in the Rust regex crate)
    return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
static int is_nl(uint32_t c) { return c == '\r' || c == '\n'; }
static uint32_t lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// Length (in code points) of the regex match starting at cps[i]; cps has n entries.
static int match_piece(const uint32_t* cps, int n, int i) {
    uint32_t c = cps[i];
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c == '\'' && i + 1 < n) {
        uint32_t a = lower(cps[i + 1]), b = i + 2 < n ? lower(cps[i + 2]) : 0;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    {
        int j = i;
        if (!is_L(c) && !is_nl(c) && !is_N(c) && i + 1 < n && is_L(cps[i + 1])) j++;
        if (is_L(cps[j])) {
            while (j < n && is_L(cps[j])) j++;
            return j - i;
        }
    }
    // \p{N}
    if (is_N(c)) return 1;
    // " ?[^\s\p{L}\p{N}]+[\r\n]*"
    {
        int j = i;
        if (c == ' ') j++;
        int k = j;
        while (k < n && !is_S(cps[k]) && !is_L(cps[k]) && !is_N(cps[k])) k++;
        if (k > j) {
            while (k < n && is_nl(cps[k])) k++;
            return k - i;
        }
    }
    if (is_S(c)) {
        int e = i;
        while (e < n && is_S(cps[e])) e++;
        // \s*[\r\n]+ : up to and including the last newline of the whitespace run
        int last_nl = -1;
        for (int k = i; k < e; k++) if (is_nl(cps[k])) last_nl = k;
        if (last_nl >= 0) return last_nl + 1 - i;
        // \s+(?!\S) : the run, minus its last character if a non-space follows
        if (e == n) return e - i;
        if (e - 1 > i) return e - 1 - i;
        // \s+
        return e - i;
    }
    return 1;
}

// ---------------------------------------------------------------- BPE

typedef struct { int32_t* v; int n, cap; } ivec;
static void ivec_push(ivec* a, int32_t x) {
    if (a->n == a->cap) { a->cap = a->cap ? 2 * a->cap : 64; a->v = realloc(a->v, a->cap * sizeof(int32_t)); }
    a->v[a->n++] = x;
}

// BPE on one pre-tokenized piece (raw bytes) -> ids appended to out.
static void bpe_piece(const tokenizer* t, const char* bytes, size_t nb, ivec* out) {
    // Byte-level string of the piece and the start offset of each symbol in it.
    char* s = malloc(2 * nb + 1);
    int* start = malloc((nb + 1) * sizeof(int));
    size_t sl = 0;
    for (size_t i = 0; i < nb; i++) {
        start[i] = (int)sl;
        const char* e = t->byte_enc[(unsigned char)bytes[i]];
        size_t el = strlen(e);
        memcpy(s + sl, e, el);
        sl += el;
    }
    s[sl] = 0;
    int nsym = (int)nb;
    start[nsym] = (int)sl;

    {
        char* key = malloc(sl + 2);
        while (nsym > 1) {
            int best = -1;
            int32_t best_rank = INT32_MAX;
            for (int k = 0; k + 1 < nsym; k++) {
                size_t la = start[k + 1] - start[k], lb = start[k + 2] - start[k + 1];
                memcpy(key, s + start[k], la);
                key[la] = ' ';
                memcpy(key + la + 1, s + start[k + 1], lb);
                int32_t r = hmap_get(&t->merges, key, la + 1 + lb);
                if (r >= 0 && r < best_rank) { best_rank = r; best = k; }
            }
            if (best < 0) break;
            memmove(start + best + 1, start + best + 2, (nsym - best - 1) * sizeof(int));
            nsym--;
        }
        free(key);
    }
    for (int k = 0; k < nsym; k++) {
        int32_t id = hmap_get(&t->vocab, s + start[k], start[k + 1] - start[k]);
        if (id < 0) fprintf(stderr, "tokenizer: no id for a BPE symbol\n");
        else ivec_push(out, id);
    }
    free(s);
    free(start);
}

static void encode_span(const tokenizer* t, const char* text, size_t n, ivec* out) {
    if (!n) return;
    uint32_t* cps = malloc(n * sizeof(uint32_t));
    size_t* off = malloc((n + 1) * sizeof(size_t));
    int ncp = 0;
    for (size_t i = 0; i < n;) {
        off[ncp] = i;
        i += utf8_decode(text + i, n - i, &cps[ncp]);
        ncp++;
    }
    off[ncp] = n;
    for (int i = 0; i < ncp;) {
        int len = match_piece(cps, ncp, i);
        bpe_piece(t, text + off[i], off[i + len] - off[i], out);
        i += len;
    }
    free(cps);
    free(off);
}

int tok_encode(const tokenizer* t, const char* text, int32_t** ids) {
    ivec out = {0};
    size_t n = strlen(text), pos = 0;
    while (pos < n) {
        // Earliest (then longest) added token in the rest of the text.
        size_t best_at = n, best_len = 0;
        int best = -1;
        for (int a = 0; a < t->n_added; a++) {
            const char* f = strstr(text + pos, t->added[a].text);
            if (!f) continue;
            size_t at = (size_t)(f - text);
            if (at < best_at || (at == best_at && t->added[a].len > best_len)) {
                best_at = at;
                best_len = t->added[a].len;
                best = a;
            }
        }
        encode_span(t, text + pos, best_at - pos, &out);
        if (best < 0) break;
        ivec_push(&out, t->added[best].id);
        pos = best_at + best_len;
    }
    *ids = out.v;
    return out.n;
}
