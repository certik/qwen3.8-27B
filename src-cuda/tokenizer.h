// Byte-level BPE tokenizer for Qwen3.5/3.8, read directly from the model's tokenizer.json.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct tokenizer tokenizer;

tokenizer* tok_load(const char* tokenizer_json_path);
void tok_free(tokenizer* t);

// Encodes UTF-8 text. Added/special tokens such as <|im_start|> that appear literally in
// the text are recognized. Returns the number of ids; *ids is malloc'ed.
int tok_encode(const tokenizer* t, const char* text, int32_t** ids);

// Returns the bytes of one token (not NUL-terminated) and its length.
const char* tok_decode(const tokenizer* t, int32_t id, size_t* len);

int tok_vocab_size(const tokenizer* t);
