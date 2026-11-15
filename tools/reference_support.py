"""Shared, versioned input/identity contract; never modifies a model directory."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_FLAGS = {"enable_thinking": False, "reasoning_effort": "low",
                  "add_generation_prompt": True}


def effective_prefill_ttft_tps(input_tokens, prefill_ttft_seconds):
    """Prompt tokens per TTFT wall second, not isolated prefill compute."""
    if type(input_tokens) is not int or input_tokens < 1:
        raise ValueError("input_tokens must be a positive integer")
    if not math.isfinite(prefill_ttft_seconds) or prefill_ttft_seconds <= 0:
        raise ValueError("prefill_ttft_seconds must be finite and positive")
    return input_tokens / prefill_ttft_seconds


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def validate_ids(ids, vocab_size):
    if not isinstance(ids, list) or not ids:
        raise ValueError("input IDs must be a nonempty JSON list")
    if any(type(x) is not int or x < 0 or x >= vocab_size for x in ids):
        raise ValueError("input ID out of vocabulary or not an integer")
    return ids


def tokenizer_size(tokenizer):
    # HF vocab_size excludes added special tokens, which this model uses in chat.
    return max(tokenizer.get_vocab().values()) + 1


def model_vocab_size(model, tokenizer):
    args = getattr(getattr(model, "language_model", model), "args", None)
    size = args.get("vocab_size") if isinstance(args, dict) else getattr(args, "vocab_size", None)
    return size if size is not None else tokenizer_size(tokenizer)


def prompt_input(tokenizer, user, ids_path=None, raw=False, thinking=False, vocab_size=None):
    flags = ({"add_generation_prompt": True} if thinking is None else
             {**TEMPLATE_FLAGS, "enable_thinking": thinking})
    bound = tokenizer_size(tokenizer) if vocab_size is None else vocab_size
    if ids_path:
        value = json.loads(Path(ids_path).read_text())
        ids = value["input_ids"] if isinstance(value, dict) else value
        return {"mode": "input_ids", "rendered_prompt": None, "template_flags": None,
                "input_ids": validate_ids(ids, bound)}
    rendered = user if raw else tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, **flags)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    return {"mode": "raw_text" if raw else "chat", "user": user,
            "rendered_prompt": rendered, "template_flags": None if raw else flags,
            "input_ids": validate_ids(ids, bound)}


def freeze_prefix(tokenizer, prompt, length):
    ids = prompt["input_ids"][:length]
    if len(ids) != length:
        raise ValueError("source prompt is shorter than the requested prefix")
    rendered = tokenizer.decode(ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    if tokenizer.encode(rendered, add_special_tokens=False) != ids:
        raise ValueError("prefix does not roundtrip exactly; use input-ID mode")
    return {**prompt, "mode": "input_ids_prefix",
            "source_text": prompt["rendered_prompt"], "rendered_prompt": rendered,
            "input_ids": ids, "truncation": f"first exactly {length} source input IDs"}


def model_identity(model_dir, cache_path):
    """Hash payloads once; reuse only for exactly matching file stat identities."""
    model_dir = Path(model_dir)
    files = sorted(model_dir.glob("*.safetensors"))
    small = ["config.json", "model.safetensors.index.json", "tokenizer.json",
             "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]
    files += [model_dir / name for name in small if (model_dir / name).exists()]
    stat = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in files}
    cache_path = Path(cache_path)
    if cache_path.exists():
        old = json.loads(cache_path.read_text())
        if old.get("files_stat") == stat and old.get("model_dir") == str(model_dir.resolve()):
            return old
    hashes = {p.name: sha256(p) for p in files}
    shards, tensors = {}, {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(length)
        header = json.loads(header_bytes)
        shards[path.name] = {"header_sha256": hashlib.sha256(header_bytes).hexdigest(),
                             "header_bytes": length}
        for name, spec in header.items():
            if name != "__metadata__":
                tensors[name] = {**spec, "shard": path.name}
    text = {k: v for k, v in tensors.items()
            if k.startswith("language_model.") or k.startswith("model.language_model.")}
    identity = {"schema": 1, "model_dir": str(model_dir.resolve()),
                "model_name": model_dir.name, "files_stat": stat, "sha256": hashes,
                "shards": shards, "tensors": tensors, "tensor_count": len(tensors),
                "text_tensor_count": len(text),
                "text_payload_bytes": sum(t["data_offsets"][1] - t["data_offsets"][0]
                                          for t in text.values()),
                "config": json.loads((model_dir / "config.json").read_text())}
    write_json(cache_path, identity)
    return identity


def runtime_identity():
    import mlx.core as mx
    import mlx_lm.models.qwen3_5 as architecture
    import mlx_lm.models.qwen3_next as blocks
    import mlx_lm.models.gated_delta as recurrence
    import mlx_lm.models.activations as activations
    # importlib avoids the public mlx_lm.generate function shadowing its module.
    import importlib
    generation = importlib.import_module("mlx_lm.generate")
    modules = [architecture, blocks, recurrence, activations, generation]
    dirty = command("git", "diff", "--binary", "HEAD")
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "tracked_diff_sha256": hashlib.sha256(dirty.encode()).hexdigest(),
        "tool_sources_sha256": {str(p.relative_to(ROOT)): sha256(p)
                               for p in sorted((ROOT / "tools").glob("*.py"))},
        "pixi_manifest_sha256": sha256(ROOT / "pixi.toml"),
        "pixi_lock_sha256": sha256(ROOT / "pixi.lock"),
        "validation_contract_sha256": sha256(ROOT / "tools/validation_contract.json"),
        "packages": {k: importlib.metadata.version(k)
                     for k in ["mlx", "mlx-metal", "mlx-lm", "numpy", "transformers", "tokenizers"]},
        "mlx_source_sha256": {Path(m.__file__).name: sha256(m.__file__) for m in modules},
        "python": platform.python_version(), "platform": platform.platform(),
        "pixi": command("pixi", "--version"),
        "clang": command("xcrun", "clang", "--version"),
        "apple_sdk_version": command("xcrun", "--show-sdk-version"),
        "c_reference_flags": "-std=c99 -O2 -ffp-contract=off; no fast-math",
        "c_build_jobs": 4,
        "os": command("sw_vers"), "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "hardware": "not collected: private hardware identifiers are excluded",
        "power": command("pmset", "-g", "batt"), "thermal": command("pmset", "-g", "therm"),
        "device": mx.device_info(),
        "gpu_busy_seconds": None,
        "gpu_busy_unavailable_reason": "no public MLX command-buffer busy-time API",
        "gpu_exclusivity": "one serialized model process; external GUI activity not instrumented",
        "allocation_copy_dispatch_counts": None,
        "telemetry_unavailable_reason": "public MLX API does not expose per-run counts",
    }


def eos_ids(model_dir, tokenizer):
    config = json.loads((Path(model_dir) / "config.json").read_text())
    eos = config.get("generation_config", {}).get("eos_token_id", config.get("eos_token_id"))
    if eos is None:
        return sorted(tokenizer.eos_token_ids)
    return eos if isinstance(eos, list) else [eos]
