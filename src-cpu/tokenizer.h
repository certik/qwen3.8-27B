// Qwen-style byte-level BPE tokenizer.
//
// The encoding algorithm is rank-based BPE (lowest token-id wins, matching
// HF tokenizers `BPE` model). The pre-tokenizer is an ASCII subset of the
// Qwen 3.5 regex (sufficient for the validation prompt and chat scaffolding).

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tokenizer_t tokenizer_t;

tokenizer_t* tk_load(const char* path, char** err);
void              tk_free(tokenizer_t*);

// Encode UTF-8 text without recognising special tokens. Special tokens
// (e.g. <|im_start|>) must be tokenised separately via tk_special.
// Output capacity must be at least `n_bytes + 1`. Returns count, or -1 on error.
int tk_encode_ordinary(const tokenizer_t* tk,
                       const char* text, size_t n_bytes,
                       int* out_ids, int max_ids);

// Look up the id of a special token by its surface form. Returns -1 if not found.
int tk_special(const tokenizer_t* tk, const char* name);

// Decode a sequence of ids into raw bytes (no UTF-8 validation).
int tk_decode(const tokenizer_t* tk,
              const int* ids, int n_ids,
              char* out, int max_bytes);

int  tk_n_vocab(const tokenizer_t*);
const char* tk_token_bytes(const tokenizer_t*, int id, int* out_len);

#ifdef __cplusplus
}
#endif
#endif
