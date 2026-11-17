"""Tokenizer tests: qwen_tokenizer.f90 against recorded HF tokenizer output.

    python src-fortran/tests/test_tokenizer.py --model DIR [--binary build/fortran-tokenizer/gfortran/test_tokenizer]
                                               [--regenerate]

Needs the model's tokenizer files (tokenizer.json, generation_config.json,
...) in DIR.  Python is stdlib-only; if `tokenizers` is importable the test
also compares randomized corpora, a long text and the chat template (via
`transformers`, when present) against the reference implementation, and
--regenerate rewrites tokenizer_cases.json from it.
"""
import argparse
import json
import pathlib
import random
import subprocess
import sys
import tempfile
import time
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
CASES = HERE / "tokenizer_cases.json"

TEXTS = [
    "", " ", "\n", "Hello, world!", "Here's a thinking process", "What is the capital of France?",
    "It's 2024. We're 'quoted' and they'd've gone, I'LL say 'S.", "x = 3.14159 + 42 * (y - 7)",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "int main(void) {\n\tprintf(\"%d\\n\", 1);\n\treturn 0;\n}\n", "!!! ??? ... --- === ### $$$ ```",
    "a  b   c    d", "trailing spaces   \nand\r\nCRLF\r\n\r\nblank lines\n\n\n", "\t\ttabs\there",
    "1234567890 0.001 1,000,000 -5 +7 3rd", "snake_case camelCase CONSTANT_CASE kebab-case",
    "https://example.com/a/b?c=d&e=f#g", "café naïve über Ångström ñ",
    "café naïve über Ångström ñ", "中文测试，你好世界。",
    "日本語のテスト", "한국어 테스트", "русский текст",
    "العربية", "हिन्दी", "emoji \U0001F600\U0001F680 and \U0001F468‍\U0001F469‍\U0001F467",
    "각 각", "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "<think>\nhm\n</think>\n\nanswer<|endoftext|>", "<|im_start|", "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>",
    "nbsp here and em space　ideographic", "­ soft hyphen, zero​width", " line sep para sep",
    "word" * 50, " " * 40 + "x", "\n" * 30, "a" * 300, "\U0001F600" * 40,
]
INVALID = [b"abc\xff\xfedef", b"\xe4\xb8", b"ok \xc0\x80 bad \xf0\x9f\x98", b"\xed\xa0\x80", b"\x80\x80\x80", b"\xf4\x90\x80\x80x"]


def run_driver(binary, model, lines, workdir):
    cases = workdir / "cases.txt"
    out = workdir / "out.txt"
    cases.write_text("".join(line + "\n" for line in lines))
    subprocess.run([str(binary), str(model), str(cases), str(out)], check=True)
    result = out.read_text().split("\n")[:-1]
    assert len(result) == len(lines), (len(result), len(lines))
    return result


def encode(binary, model, texts, workdir):
    lines = ["E " + (t if isinstance(t, bytes) else t.encode()).hex() for t in texts]
    return [[int(x) for x in r.split()] for r in run_driver(binary, model, lines, workdir)]


def decode(binary, model, id_lists, workdir):
    lines = ["D " + " ".join(map(str, ids)) for ids in id_lists]
    return [bytes.fromhex(r) for r in run_driver(binary, model, lines, workdir)]


def chat_lines(prompts):
    return ["C %s %s %s %s" % ("think" if think else "nothink", effort, user.encode().hex(), system.encode().hex())
            for user, system, think, effort in prompts]


CHATS = [
    ("What is the capital of France?", "", True, "xhigh"),
    ("What is the capital of France?", "", False, "xhigh"),
    ("What is the capital of France?", "", True, "medium"),
    ("What is the capital of France?", "", True, "low"),
    ("  padded\n\n", "You are terse.  \n", True, "xhigh"),
    ("Hi <|im_end|> there", "Be nice.", False, "xhigh"),
    ("Hi", "Be nice.", True, "low"),
    ("Hi", "   ", False, "xhigh"),
]


def reference_chat(model, user, system, think, effort):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(model))
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    kwargs = dict(add_generation_prompt=True, tokenize=False, enable_thinking=think)
    if think:
        kwargs["reasoning_effort"] = effort
    return tk.apply_chat_template(msgs, **kwargs)


def regenerate(model):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(model / "tokenizer.json"))
    data = {"texts": [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS],
            "decode": [{"ids": [8160, 579, 264, 7047, 1817], "text": "Here's a thinking process"}],
            "chat": []}
    for user, system, think, effort in CHATS:
        text = reference_chat(model, user, system, think, effort)
        data["chat"].append({"user": user, "system": system, "think": think, "effort": effort, "text": text,
                             "ids": tk.encode(text, add_special_tokens=False).ids})
    CASES.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=pathlib.Path)
    ap.add_argument("--binary", default="build/fortran-tokenizer/gfortran/test_tokenizer", type=pathlib.Path)
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--fuzz", type=int, default=3000)
    args = ap.parse_args()
    if args.regenerate:
        regenerate(args.model)
        return
    work = pathlib.Path(tempfile.mkdtemp(prefix="tok-test-"))
    failures = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else ": " + detail))
        if not ok:
            failures.append(name)

    data = json.loads(CASES.read_text())
    # 1. recorded reference ids
    got = encode(args.binary, args.model, [c["text"] for c in data["texts"]], work)
    for c, ids in zip(data["texts"], got):
        check("ids %r" % c["text"][:30], ids == c["ids"], "%s != %s" % (ids[:20], c["ids"][:20]))
    # 2. detokenization of recorded ids and round trip (decode(encode(s)) == NFC(s))
    for c in data["decode"]:
        out, = decode(args.binary, args.model, [c["ids"]], work)
        check("decode %r" % c["text"], out == c["text"].encode(), repr(out))
    back = decode(args.binary, args.model, [c["ids"] for c in data["texts"]], work)
    for c, text in zip(data["texts"], back):
        check("roundtrip %r" % c["text"][:30], text == unicodedata.normalize("NFC", c["text"]).encode(), repr(text[:60]))
    # 3. invalid UTF-8 in -> U+FFFD (python's replace semantics)
    for raw in INVALID:
        ids, = encode(args.binary, args.model, [raw], work)
        text, = decode(args.binary, args.model, [ids], work)
        check("invalid %r" % raw, text == raw.decode("utf-8", "replace").encode(), repr(text))
    # 4. bytes split over tokens are buffered until complete; a truncated or
    #    broken sequence becomes U+FFFD exactly as bytes.decode(errors="replace")
    vocab = json.loads((args.model / "vocab.json").read_text())
    keep = [b for b in range(256) if 33 <= b <= 126 or 161 <= b <= 172 or b >= 174]
    b2u = {b: chr(b) for b in keep}
    b2u.update({b: chr(256 + i) for i, b in enumerate(x for x in range(256) if x not in keep)})
    byte_id = [vocab[b2u[b]] for b in range(256)]
    rng = random.Random(7)
    seqs = [list(smile) for smile in ("\U0001F600".encode(), "\u00e9".encode(), "\u20ac".encode())]
    seqs += [[0xf0, 0x9f], [0xf0, 0x9f, 0x61], [0xe2, 0x82], [0xc3], [0xc3, 0xc3, 0xa9], [0xed, 0xa0, 0x80]]
    seqs += [[rng.choice([0x61, 0x80, 0x9f, 0xbf, 0xc2, 0xc3, 0xe0, 0xe2, 0xed, 0xf0, 0xf4, 0xff, 0xa9, 0x82])
              for _ in range(rng.randint(1, 10))] for _ in range(500)]
    outs = decode(args.binary, args.model, [[byte_id[b] for b in seq] for seq in seqs], work)
    bad = [seq for seq, o in zip(seqs, outs) if o != bytes(seq).decode("utf-8", "replace").encode()]
    check("byte tokens split across characters (%d sequences)" % len(seqs), not bad, repr(bad[:3]))
    # 5. chat template
    for case, line in zip(data["chat"], run_driver(args.binary, args.model, chat_lines(
            [(c["user"], c["system"], c["think"], c["effort"]) for c in data["chat"]]), work)):
        text_hex, ids = line.split(" ", 1)
        text = bytes.fromhex(text_hex).decode()
        check("chat text %r %s" % (case["user"][:20], case["effort"] if case["think"] else "nothink"),
              text == case["text"], repr(text))
        check("chat ids %r" % case["user"][:20], [int(x) for x in ids.split()] == case["ids"])
    # 6. NFC examples
    nfc = {"é": "é", "ậ": "ậ", "각": "각", "Å": "Å", "q̣̇": "q̣̇"}
    outs = run_driver(args.binary, args.model, ["N " + k.encode().hex() for k in nfc], work)
    for (k, v), o in zip(nfc.items(), outs):
        check("nfc %r" % k, bytes.fromhex(o) == v.encode(), o)
    # 7. long input (100 KB) must be fast and round-trip
    long_text = ("The quick brown fox jumps over the lazy dog. " * 2200)[:100000]
    start = time.time()
    ids, = encode(args.binary, args.model, [long_text], work)
    elapsed = time.time() - start
    text, = decode(args.binary, args.model, [ids], work)
    check("long text roundtrip (%d ids, %.2f s incl. load)" % (len(ids), elapsed), text == long_text.encode() and elapsed < 30)
    # 8. optional differential tests against the reference implementation
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("SKIP differential tests (tokenizers not installed)")
        Tokenizer = None
    if Tokenizer is not None:
        tk = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        rng = random.Random(20240601)
        pieces = [c for t in TEXTS for c in [t]] + ["'s", "'RE", "\r\n", "  ", "́", " ", " ", "<|im_end|>", "123"]
        fuzz = ["".join(rng.choice(pieces) for _ in range(rng.randint(0, 6))) for _ in range(args.fuzz)]
        fuzz += ["".join(chr(rng.choice([rng.randint(32, 126), rng.randint(160, 0x2fff), rng.randint(0x3000, 0xd7ff),
                                         rng.randint(0xe000, 0x1ffff)])) for _ in range(rng.randint(1, 12))) for _ in range(args.fuzz)]
        fuzz.append(long_text)
        mine = encode(args.binary, args.model, fuzz, work)
        bad = [s for s, ids in zip(fuzz, mine) if ids != tk.encode(s, add_special_tokens=False).ids]
        check("fuzz vs tokenizers (%d strings)" % len(fuzz), not bad, repr(bad[:2]))
        scan = [chr(c) for c in range(0x110000) if not 0xd800 <= c < 0xe000]
        mine = encode(args.binary, args.model, scan, work)
        bad = [s for s, ids in zip(scan, mine) if ids != tk.encode(s, add_special_tokens=False).ids]
        check("every code point vs tokenizers", not bad, repr(bad[:5]))
        n_vocab = tk.get_vocab_size(with_added_tokens=True)
        id_lists = [[rng.randrange(n_vocab) for _ in range(rng.randint(1, 12))] for _ in range(500)]
        dec = decode(args.binary, args.model, id_lists, work)
        bad = [ids for d, ids in zip(dec, id_lists) if d != tk.decode(ids, skip_special_tokens=False).encode()]
        check("decode vs tokenizers (%d random id lists)" % len(id_lists), not bad, repr(bad[:2]))
    print("%d failure(s)" % len(failures))
    sys.exit(1 if failures else 0)


main()
