#include "qwen_chat.h"

#include <stdio.h>
#include <string.h>

static int emit_literal(const tokenizer_t* tk,
                        const char* text,
                        int* out, int max_ids, int n)
{
    int got = tk_encode_ordinary(tk, text, strlen(text),
                                 out + n, max_ids - n);
    if (got < 0) return -1;
    return n + got;
}

static int emit_special(const tokenizer_t* tk, const char* name,
                        int* out, int max_ids, int n)
{
    int sid = tk_special(tk, name);
    if (sid < 0 || n >= max_ids) return -1;
    out[n] = sid;
    return n + 1;
}

int qwen_chat_build_prompt(const tokenizer_t* tk,
                           const char* system,
                           const char* user,
                           int* out_ids, int max_ids)
{
    int n = 0;

    if (system && system[0]) {
        n = emit_special(tk, "<|im_start|>", out_ids, max_ids, n);
        if (n < 0) return -1;
        n = emit_literal(tk, "system\n", out_ids, max_ids, n);
        if (n < 0) return -1;
        n = emit_literal(tk, system, out_ids, max_ids, n);
        if (n < 0) return -1;
        n = emit_special(tk, "<|im_end|>", out_ids, max_ids, n);
        if (n < 0) return -1;
        n = emit_literal(tk, "\n", out_ids, max_ids, n);
        if (n < 0) return -1;
    }

    n = emit_special(tk, "<|im_start|>", out_ids, max_ids, n);
    if (n < 0) return -1;
    n = emit_literal(tk, "user\n", out_ids, max_ids, n);
    if (n < 0) return -1;
    n = emit_literal(tk, user, out_ids, max_ids, n);
    if (n < 0) return -1;
    n = emit_special(tk, "<|im_end|>", out_ids, max_ids, n);
    if (n < 0) return -1;
    n = emit_literal(tk, "\n", out_ids, max_ids, n);
    if (n < 0) return -1;
    n = emit_special(tk, "<|im_start|>", out_ids, max_ids, n);
    if (n < 0) return -1;
    n = emit_literal(tk, "assistant\n", out_ids, max_ids, n);
    if (n < 0) return -1;
    n = emit_special(tk, "<think>", out_ids, max_ids, n);
    if (n < 0) return -1;
    n = emit_literal(tk, "\n", out_ids, max_ids, n);
    if (n < 0) return -1;

    return n;
}
