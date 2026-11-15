#include "json.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char* p; const char* end; int err; } parser;

static void fail(parser* ps, const char* what) {
    if (!ps->err) fprintf(stderr, "json: %s\n", what);
    ps->err = 1;
}

static void skip_ws(parser* ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\n' || *ps->p == '\r' || *ps->p == '\t')) ps->p++;
}

static int hex4(parser* ps, uint32_t* out) {
    if (ps->end - ps->p < 4) return 0;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = ps->p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else return 0;
    }
    ps->p += 4;
    *out = v;
    return 1;
}

static size_t put_utf8(char* o, uint32_t cp) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

// Parses a string starting at the opening quote. Decoded output is never longer than the input.
static char* parse_string(parser* ps, size_t* len) {
    ps->p++;   // opening quote
    const char* s = ps->p;
    while (ps->p < ps->end && *ps->p != '"') ps->p += (*ps->p == '\\') ? 2 : 1;
    if (ps->p >= ps->end) { fail(ps, "unterminated string"); return NULL; }
    const char* e = ps->p;
    ps->p++;   // closing quote
    char* out = malloc((size_t)(e - s) + 1);
    size_t n = 0;
    parser q = {s, e, 0};
    while (q.p < q.end) {
        char c = *q.p++;
        if (c != '\\') { out[n++] = c; continue; }
        char k = *q.p++;
        switch (k) {
        case '"': out[n++] = '"'; break;
        case '\\': out[n++] = '\\'; break;
        case '/': out[n++] = '/'; break;
        case 'b': out[n++] = '\b'; break;
        case 'f': out[n++] = '\f'; break;
        case 'n': out[n++] = '\n'; break;
        case 'r': out[n++] = '\r'; break;
        case 't': out[n++] = '\t'; break;
        case 'u': {
            uint32_t cp;
            if (!hex4(&q, &cp)) { fail(ps, "bad \\u escape"); free(out); return NULL; }
            if (cp >= 0xD800 && cp < 0xDC00 && q.end - q.p >= 6 && q.p[0] == '\\' && q.p[1] == 'u') {
                q.p += 2;
                uint32_t lo;
                if (!hex4(&q, &lo)) { fail(ps, "bad \\u escape"); free(out); return NULL; }
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
            n += put_utf8(out + n, cp);
            break;
        }
        default: fail(ps, "bad escape"); free(out); return NULL;
        }
    }
    out[n] = 0;
    *len = n;
    return out;
}

static void parse_value(parser* ps, jval* v);

typedef struct { jval* items; char** keys; size_t n, cap; } vec;

static void vec_push(vec* a, jval* item, char* key, int with_keys) {
    if (a->n == a->cap) {
        a->cap = a->cap ? 2 * a->cap : 4;
        a->items = realloc(a->items, a->cap * sizeof(jval));
        if (with_keys) a->keys = realloc(a->keys, a->cap * sizeof(char*));
    }
    a->items[a->n] = *item;
    if (with_keys) a->keys[a->n] = key;
    a->n++;
}

static void parse_container(parser* ps, jval* v, int is_obj) {
    char close = is_obj ? '}' : ']';
    vec a = {0};
    ps->p++;
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == close) { ps->p++; goto done; }
    for (;;) {
        char* key = NULL;
        skip_ws(ps);
        if (is_obj) {
            size_t kl;
            if (ps->p >= ps->end || *ps->p != '"') { fail(ps, "expected key"); break; }
            key = parse_string(ps, &kl);
            if (!key) break;
            skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') { fail(ps, "expected ':'"); free(key); break; }
            ps->p++;
        }
        jval item;
        parse_value(ps, &item);
        if (ps->err) { free(key); break; }
        vec_push(&a, &item, key, is_obj);
        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
        if (ps->p < ps->end && *ps->p == close) { ps->p++; break; }
        fail(ps, "expected ',' or closing bracket");
        break;
    }
done:
    v->type = is_obj ? J_OBJ : J_ARR;
    v->items = a.items;
    v->keys = a.keys;
    v->len = a.n;
}

static void parse_value(parser* ps, jval* v) {
    memset(v, 0, sizeof(*v));
    skip_ws(ps);
    if (ps->p >= ps->end) { fail(ps, "unexpected end"); return; }
    char c = *ps->p;
    if (c == '{') parse_container(ps, v, 1);
    else if (c == '[') parse_container(ps, v, 0);
    else if (c == '"') { v->type = J_STR; v->str = parse_string(ps, &v->len); }
    else if (!strncmp(ps->p, "true", 4)) { v->type = J_BOOL; v->num = 1; ps->p += 4; }
    else if (!strncmp(ps->p, "false", 5)) { v->type = J_BOOL; v->num = 0; ps->p += 5; }
    else if (!strncmp(ps->p, "null", 4)) { v->type = J_NULL; ps->p += 4; }
    else if (c == '-' || (c >= '0' && c <= '9')) {
        char buf[64];
        size_t n = 0;
        while (ps->p < ps->end && n < 63 && strchr("+-0123456789.eE", *ps->p)) buf[n++] = *ps->p++;
        buf[n] = 0;
        v->type = J_NUM;
        v->num = strtod(buf, NULL);
    } else fail(ps, "unexpected character");
}

jval* json_parse(const char* text, size_t n) {
    parser ps = {text, text + n, 0};
    jval* v = malloc(sizeof(jval));
    parse_value(&ps, v);
    if (ps.err) { json_free(v); return NULL; }
    return v;
}

static void free_children(jval* v) {
    if (v->type == J_STR) free(v->str);
    if (v->type == J_ARR || v->type == J_OBJ) {
        for (size_t i = 0; i < v->len; i++) {
            free_children(&v->items[i]);
            if (v->keys) free(v->keys[i]);
        }
        free(v->items);
        free(v->keys);
    }
}

void json_free(jval* v) {
    if (!v) return;
    free_children(v);
    free(v);
}

const jval* json_get(const jval* obj, const char* key) {
    if (!obj || obj->type != J_OBJ) return NULL;
    for (size_t i = 0; i < obj->len; i++)
        if (!strcmp(obj->keys[i], key)) return &obj->items[i];
    return NULL;
}

const jval* json_at(const jval* arr, size_t i) {
    if (!arr || arr->type != J_ARR || i >= arr->len) return NULL;
    return &arr->items[i];
}
