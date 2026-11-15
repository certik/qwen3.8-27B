// Minimal JSON DOM parser (enough for config.json, tokenizer.json and safetensors headers).
#pragma once
#include <stddef.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype type;
    double num;           // J_NUM, J_BOOL (0/1)
    char* str;            // J_STR: UTF-8, NUL-terminated (may contain no NULs)
    size_t len;           // J_STR: byte length; J_ARR/J_OBJ: number of children
    struct jval* items;   // J_ARR/J_OBJ children
    char** keys;          // J_OBJ keys
} jval;

// Parses text[0..n). Returns NULL on error (message on stderr). Free with json_free.
jval* json_parse(const char* text, size_t n);
void json_free(jval* v);
const jval* json_get(const jval* obj, const char* key);   // NULL if absent / not an object
const jval* json_at(const jval* arr, size_t i);           // NULL if out of range
