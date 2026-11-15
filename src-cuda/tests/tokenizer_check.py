# Compares ../qwen's tokenizer with the HuggingFace `tokenizers` library on tricky strings.
#   python tokenizer_check.py [MODEL_DIR]      (needs: pip install tokenizers)
import os, subprocess, sys
from tokenizers import Tokenizer

model = sys.argv[1] if len(sys.argv) > 1 else "/tmp/certik/model"
qwen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen")
ref = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
cases = [
    "Hello, who are you?",
    "<|im_start|>user\nWrite a short story about a robot.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "I'm here; they'll go, we've DONE it, she'd say. You're 'quoted' THE'S",
    "Numbers: 12345 3.14159 -42 1,000,000 and 2026-11-14",
    "  leading spaces\n\n\ttabs\t and    multiple   spaces   \n  \r\n end  ",
    "def foo(x):\n    return x**2 + 1  # comment\n\n\nprint(foo(3))\n",
    "Unicode: café naïve Ελληνικά русский 中文字符 日本語 한국어 emoji 🙂👍🏽 – — … «»",
    "Math: ∑_{i=1}^n i² = n(n+1)/2, α≥β, x→∞ ½ ¾ ①②",
    "URL https://example.com/a?b=c&d=e#frag, email a.b@c.de, path /usr/local/bin",
    "!!!???... ---=== $$$ %%% ^^^ &&& *** ((()))",
    "a\n\nb\n \nc \n",
    "trailing space ",
    "x   ",
    "Fortran: real(dp), intent(in) :: a(:,:)\n  do i = 1, n\n     y(i) = sum(a(:,i)*x)\n  end do",
]
bad = 0
for c in cases:
    want = ref.encode(c, add_special_tokens=False).ids
    out = subprocess.run([qwen, "-m", model, "--raw", "--tokenize", "-p", c], capture_output=True, text=True, check=True)
    got = [int(x) for x in out.stdout.split()]
    ok = got == want
    bad += not ok
    print("OK " if ok else "BAD", f"{len(want):3d} tokens", repr(c[:50]))
    if not ok:
        print("   want", want)
        print("   got ", got)
print(f"{len(cases) - bad}/{len(cases)} match")
sys.exit(1 if bad else 0)
