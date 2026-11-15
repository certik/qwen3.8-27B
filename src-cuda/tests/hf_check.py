# Teacher-forced check of ../qwen against HuggingFace transformers (bf16).
# Runs `qwen --ids` greedily, then feeds prompt + generated tokens through the HF model in one
# forward pass and reports, for every generated token, whether it is HF's top-1 choice and, if not,
# the logit gap. Needs torch + transformers and a bf16 HF checkpoint of the same weights
# (made with ../../scripts/mlx_to_hf.py; text-only config).
#   python hf_check.py HF_MODEL_DIR "prompt" [n_tokens]
import os, re, subprocess, sys
import torch
from transformers import AutoModelForCausalLM

hf_dir, prompt = sys.argv[1], sys.argv[2]
n = sys.argv[3] if len(sys.argv) > 3 else "200"
qwen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen")
run = subprocess.run([qwen, "-p", prompt, "--no-think", "-n", n, "--ids"], capture_output=True, text=True, check=True)
ids = lambda tag: [int(x) for x in re.search(tag + r" ids \(\d+\):([ \d]*)", run.stderr).group(1).split()]
p, g = ids("prompt"), ids("output")
print(run.stderr.strip().splitlines()[-1])

m = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    logits = m(torch.tensor([p + g[:-1]], device="cuda")).logits[0, len(p) - 1:].float()
top = logits.argmax(-1).tolist()
agree = sum(a == b for a, b in zip(top, g))
print(f"{agree}/{len(g)} generated tokens are HF's top-1 choice")
for i, (a, b) in enumerate(zip(top, g)):
    if a != b:
        print(f"  step {i}: qwen {b} (HF logit {logits[i, b]:.3f}), HF top {a} ({logits[i, a]:.3f}), gap {logits[i, a] - logits[i, b]:.3f}")
