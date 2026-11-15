#ifndef QWEN_CHAT_H
#define QWEN_CHAT_H

#include "tokenizer.h"

// Build prompt ids for Qwen 3.5 chat template:
//
//   <|im_start|>user\n<user_text><|im_end|>\n<|im_start|>assistant\n<think>\n
//
// (Matches HF tokenizer.apply_chat_template([{"role":"user","content":user}],
//  add_generation_prompt=True). The system message is ignored if NULL/empty;
//  for Qwen 3.5 there's no default system prompt unless tools are declared.)
//
// Returns the number of token ids written, or -1 if capacity is exceeded.
int qwen_chat_build_prompt(const tokenizer_t* tk,
                           const char* system,   // may be NULL
                           const char* user,
                           int* out_ids, int max_ids);

#endif
