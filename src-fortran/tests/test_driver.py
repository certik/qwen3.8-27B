"""Small deterministic dense-hybrid HF fixtures; unchanged C is the only math oracle.

Python is test-only stdlib. No model downloads, MLX, GPU, NumPy or C linkage
in the Fortran programs. Every mismatch reports the first C-schedule operation.
"""
import copy
import json
import math
import pathlib
import random
import resource
import shutil
import struct
import subprocess
import sys

from test_model_io import config_fixture, write_archive, write_json


ROOT = pathlib.Path("build/fortran-driver")
COMPILERS = ("gfortran", "flang")
C_ORACLE = ROOT / "qwen-c-oracle"
OPERATIONS = (
    "x_in", "h_after_input_norm", "qkv", "z", "a", "b", "conv_state_in",
    "delta_state_in", "conv_out", "q_lin", "k_lin", "v_lin", "q_norm_lin",
    "k_norm_lin", "g", "beta", "delta_y", "delta_state", "norm_out",
    "q_proj", "k_proj", "v_proj", "gate", "q_after_qnorm", "k_after_knorm",
    "q_after_rope", "k_after_rope", "attn", "out_oproj",
    "x_after_attn_residual", "h_after_post_norm", "gate_proj", "up_proj",
    "swiglu_mid", "dense_down", "out_mlp", "x_after_mlp_residual",
    "cache_conv", "cache_delta", "cache_k", "cache_v",
)


def bf16(value):
    word, = struct.unpack("<I", struct.pack("<f", value))
    return struct.pack("<H", (word + 0x7fff + ((word >> 16) & 1)) >> 16)


def fixture(directory, head_dim):
    cfg = config_fixture()
    tc = cfg["text_config"]
    tc.update(num_attention_heads=3, num_key_value_heads=1, head_dim=head_dim,
              linear_num_key_heads=1, linear_num_value_heads=3,
              linear_key_head_dim=128, linear_value_head_dim=128)
    tc["rope_parameters"]["mrope_section"] = [head_dim // 16, head_dim // 32, head_dim // 32]
    entries = []
    rng = random.Random(3861 + head_dim)

    def vector(name, n, kind="norm"):
        if kind == "norm":
            values = [0.8 + rng.random() * 0.4 for _ in range(n)]
        elif kind == "alog":
            values = [-1.4 + rng.random() * 0.6 for _ in range(n)]
        else:
            values = [-0.3 + rng.random() * 0.6 for _ in range(n)]
        entries.append((name, "BF16", [n], b"".join(map(bf16, values))))

    def q8(name, k, n):
        # Vary packed signs/bytes, rows, affine groups and offsets. These are
        # nonzero model weights, not zero-path or near-tie-MLX acceptance tests.
        words = bytes(rng.randrange(256) for _ in range(k * n))
        scale = 0.002 if "embed_tokens" in name else 0.0007
        scales = [scale * (0.8 + 0.4 * rng.random()) for _ in range(n * k // 64)]
        offsets = [-128 * s + (rng.random() - 0.5) * 0.01 for s in scales]
        entries.extend([
            (name + ".weight", "U32", [n, k // 4], words),
            (name + ".scales", "BF16", [n, k // 64], b"".join(map(bf16, scales))),
            (name + ".biases", "BF16", [n, k // 64], b"".join(map(bf16, offsets))),
        ])

    q8("language_model.model.embed_tokens", 64, 128)
    q8("language_model.lm_head", 64, 128)
    vector("language_model.model.norm.weight", 64)
    for li in range(4):
        p = f"language_model.model.layers.{li}"
        vector(p + ".input_layernorm.weight", 64)
        vector(p + ".post_attention_layernorm.weight", 64)
        for name, k, n in [("gate_proj", 64, 128), ("up_proj", 64, 128), ("down_proj", 128, 64)]:
            q8(p + ".mlp." + name, k, n)
        if li != 3:
            p += ".linear_attn"
            for name, k, n in [("in_proj_qkv", 64, 640), ("in_proj_z", 64, 384),
                               ("in_proj_b", 64, 3), ("in_proj_a", 64, 3), ("out_proj", 384, 64)]:
                q8(p + "." + name, k, n)
            conv = [(-0.15 + rng.random() * 0.3) for _ in range(640 * 4)]
            entries.append((p + ".conv1d.weight", "BF16", [640, 4, 1], b"".join(map(bf16, conv))))
            vector(p + ".A_log", 3, "alog")
            vector(p + ".dt_bias", 3, "bias")
            vector(p + ".norm.weight", 128)
        else:
            p += ".self_attn"
            for name, k, n in [("q_proj", 64, 6 * head_dim), ("k_proj", 64, head_dim),
                               ("v_proj", 64, head_dim), ("o_proj", 3 * head_dim, 64)]:
                q8(p + "." + name, k, n)
            vector(p + ".q_norm.weight", head_dim)
            vector(p + ".k_norm.weight", head_dim)
    # Several real shards, without a model-sized resident payload.
    assert sum(len(e[3]) for e in entries) < 4 * 1024 * 1024
    write_archive(directory, [entries[::2], entries[1::2]], cfg)
    return cfg


def execute(binary, args, log, success=True):
    with log.open("wb") as stream:
        result = subprocess.run([str(binary), *map(str, args)], stdout=subprocess.PIPE, stderr=stream)
        stream.write(result.stdout)
    if (result.returncode == 0) != success:
        raise AssertionError(f"{binary}: unexpected status {result.returncode}; see {log}\n{log.read_text()[-4000:]}")
    return result.stdout.decode() if success else log.read_text()


def tensor(path):
    raw = path.read_bytes()
    assert raw[:8] == b"LLMTNSR1", path
    dtype, rank = struct.unpack_from("<II", raw, 8)
    shape = struct.unpack_from("<" + "q" * rank, raw, 16)
    assert dtype == 0 and len(raw) == 16 + 8 * rank + 4 * math.prod(shape), path
    return shape, raw[16 + 8 * rank:]


def ordered_names():
    yield "x_after_embed.bin"
    for li in range(4):
        for name in OPERATIONS:
            yield f"{name}_{li}.bin"
    yield "x_after_final_norm.bin"
    yield "logits.bin"


def compare_call(expected, actual, counts):
    left = {p.name for p in expected.glob("*.bin")}
    right = {p.name for p in actual.glob("*.bin")}
    assert left == right, (expected, left ^ right)
    for name in ordered_names():
        if name not in left:
            continue
        es, ep = tensor(expected / name)
        fs, fp = tensor(actual / name)
        assert es == fs, (actual / name, es, fs)
        if ep != fp:
            for index in range(len(ep) // 4):
                a, b = ep[index * 4:index * 4 + 4], fp[index * 4:index * 4 + 4]
                if a != b:
                    raise AssertionError(f"first differing operation: {actual / name}, element={index}, "
                                         f"C={a.hex()} Fortran={b.hex()}")
        elements = len(ep) // 4
        counts["tensors"] += 1
        counts["elements"] += elements
        if name.startswith(("cache_conv", "cache_delta", "cache_k_", "cache_v_")):
            counts["state_elements"] += elements
            counts["state_tensors"] += 1
        if name.startswith("cache_delta"):
            counts["fp32_state_elements"] += elements
        if name.startswith("cache_conv"):
            counts["conv_state_elements"] += elements
        if name.startswith(("cache_k_", "cache_v_")):
            counts["kv_state_elements"] += elements
        if name == "logits.bin":
            counts["logits"] += elements


def prepare_dump(path, steps):
    path.mkdir()
    for step in range(1, steps):
        (path / f"decode-{step:03d}").mkdir()
    (path / "reset").mkdir()


def witnesses(path):
    # Require actual nonzero recurrence/carry/cache and distinct interleaved
    # Q/gate blocks; prevent vacuous synthetic coverage.
    for name in ["cache_delta_0", "cache_conv_0", "cache_k_3", "cache_v_3", "g_0", "beta_0"]:
        _, payload = tensor(path / f"{name}.bin")
        assert any(payload), name
    _, qp = tensor(path / "q_proj_3.bin")
    _, gate = tensor(path / "gate_3.bin")
    assert qp[:len(gate)] != gate
    for name in ("gate_proj_0", "up_proj_0", "conv_out_0", "norm_out_0"):
        assert any(tensor(path / f"{name}.bin")[1]), name


def main():
    ROOT.mkdir(exist_ok=True, parents=True)
    # Retain small fixtures/dumps/logs until the next run of this owned test.
    run = ROOT / "checks"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir()
    models = {dim: (run / f"model-{dim}") for dim in (64, 256)}
    configs = {dim: fixture(path, dim) for dim, path in models.items()}
    all_counts = {}
    cases = [
        ("matrix-full", 64, 19, 17, True),
        ("matrix-fallback-long", 256, 67, 17, True),
        ("vector-greedy", 256, 3, 17, False),
        ("single-greedy", 64, 1, 17, False),
    ]
    for compiler in COMPILERS:
        all_counts[compiler] = dict(tensors=0, elements=0, state_elements=0, state_tensors=0,
                                    fp32_state_elements=0, conv_state_elements=0, kv_state_elements=0,
                                    logits=0, reset_elements=0, reset_tensors=0, cases=0, rejects=0,
                                    positions=0, generated_ids=0, consumed_ids=0)
        output = execute(ROOT / compiler / "test_driver_api", ["state-bits"],
                         run / f"{compiler}-state-bits.log")
        assert output.strip() == "PASS: 43 exact reset-comparison cases (signed zero and quiet NaN payloads)"
        print(f"PASS {compiler}: 43 scalar-bit reset comparison cases", flush=True)
    for name, dim, prompt_len, steps, teacher in cases:
        case = run / name
        case.mkdir()
        inp = [((i * 37 + 11) % 128) for i in range(prompt_len)]
        teacher_ids = [(i * 13 + 7) % 128 for i in range(steps - 1)]
        input_path = write_json(case / "input.json", inp)
        teacher_path = write_json(case / "teacher.json", teacher_ids)
        common = ["--model", models[dim], "--input-ids", input_path, "--max-tokens", steps,
                  "--fixed-steps", "--reset-check"]
        if teacher:
            common += ["--teacher-ids", teacher_path]
        c_dump, c_result = case / "c", case / "c.json"
        execute(C_ORACLE, [*common, "--dump", c_dump, "--result", c_result], case / "c.log")
        c_report = json.loads(c_result.read_text())
        assert c_report["actual_generated"] == steps
        assert c_report["inference_cache_tokens_processed"] == prompt_len + steps - 1
        assert c_report["reset_cache_and_logits_exact"]
        assert c_report["consumed_decode_ids"] == (teacher_ids if teacher else c_report["generated_ids"][:-1])
        witnesses(c_dump)
        for compiler in COMPILERS:
            binary = ROOT / compiler / "qwen-raw"
            dump, result = case / compiler, case / f"{compiler}.json"
            prepare_dump(dump, steps)
            execute(binary, [*common, "--dump", dump, "--result", result], case / f"{compiler}.log")
            report = json.loads(result.read_text())
            for key, value in report.items():
                assert c_report[key] == value, (name, compiler, key, c_report[key], value)
            counts = all_counts[compiler]
            compare_call(c_dump, dump, counts)
            assert tensor(dump / "cache_k_3.bin")[0][0] == prompt_len
            for step in range(1, steps):
                compare_call(c_dump / f"decode-{step:03d}", dump / f"decode-{step:03d}", counts)
                assert tensor(dump / f"decode-{step:03d}" / "cache_k_3.bin")[0][0] == prompt_len + step
            before_elements, before_tensors = counts["elements"], counts["tensors"]
            compare_call(c_dump / "reset", dump / "reset", counts)
            counts["reset_elements"] += counts["elements"] - before_elements
            counts["reset_tensors"] += counts["tensors"] - before_tensors
            # Every dumped prefill operation, not just logits, is byte-exact
            # after resetting a model that has performed all 16 decode calls.
            for p in dump.glob("*.bin"):
                assert p.read_bytes() == (dump / "reset" / p.name).read_bytes(), p
            counts["cases"] += 1
            counts["positions"] += steps + 1
            counts["generated_ids"] += steps
            counts["consumed_ids"] += steps - 1
        print(f"PASS {name}: prefill={prompt_len} generated={steps} decode=16 reset=1; both compilers", flush=True)

    # EOS must stop immediately, fixed steps must ignore it, and there is no
    # extra unemitted forward. Reuse an existing model with an EOS-only config
    # change, then restore the original config before subsequent checks.
    config_path = models[64] / "config.json"
    cfg = copy.deepcopy(configs[64])
    first = json.loads((run / "single-greedy" / "c.json").read_text())["generated_ids"][0]
    cfg["eos_token_id"] = first
    write_json(config_path, cfg)
    input_path = run / "single-greedy" / "input.json"
    for compiler in COMPILERS:
        binary = ROOT / compiler / "qwen-raw"
        log = run / f"{compiler}-eos.log"
        report = json.loads(execute(binary, ["--model", models[64], "--input-ids", input_path,
                                              "--max-tokens", 17], log))
        assert report["generated_ids"] == [first]
        assert report["consumed_decode_ids"] == []
        assert report["inference_forward_calls"] == report["inference_cache_tokens_processed"] == 1
        fixed = json.loads(execute(binary, ["--model", models[64], "--input-ids", input_path,
                                               "--max-tokens", 3, "--fixed-steps"], run / f"{compiler}-fixed.log"))
        assert fixed["actual_generated"] == 3 and fixed["inference_forward_calls"] == 3
    write_json(config_path, configs[64])

    # The Fortran raw-ID reader also accepts an object; C uses only arrays.
    object_ids = write_json(run / "object-ids.json", {"input_ids": [11]})
    for compiler in COMPILERS:
        binary = ROOT / compiler / "qwen-raw"
        report = json.loads(execute(binary, ["--model", models[64], "--input-ids", object_ids,
                                              "--max-tokens", 1], run / f"{compiler}-object.log"))
        assert report["input_ids"] == [11] and report["generated_ids"] == [first]
        metadata = json.loads(execute(binary, ["--model", models[64], "--inspect"],
                                     run / f"{compiler}-inspect.log"))
        assert metadata["metadata_only"] and metadata["text_tensors"] > 0
        config = json.loads(execute(binary, ["--model", models[64], "--check-config"],
                                   run / f"{compiler}-check-config.log"))
        assert config == {"config_valid": True, "dense_hybrid": True}
        padded = run / f"{compiler}-padded-dump"
        padded.mkdir()
        execute(ROOT / compiler / "test_driver_api", ["padded-dump", models[64], padded],
                run / f"{compiler}-padded-dump.log")
        original = run / "single-greedy" / "c"
        assert {p.name for p in padded.glob("*.bin")} == {p.name for p in original.glob("*.bin")}
        for p in padded.glob("*.bin"):
            assert p.read_bytes() == (original / p.name).read_bytes(), p

    # Rejections must fail before model payload selection or any inference.
    bad_files = {
        "negative": ([-1], "outside vocabulary"),
        "vocab": ([128], "outside vocabulary"),
        "fractional": ([1.5], "integer"),
        "empty": ([], "empty"),
        "nested": ([[1]], "type"),
    }
    bad_args = []
    for name, (value, diagnostic) in bad_files.items():
        path = write_json(run / f"bad-{name}.json", value)
        bad_args.append((name, ["--input-ids", path], diagnostic))
    bad_args += [
        ("capacity", ["--input-ids", input_path, "--max-tokens", 513], "capacity"),
        ("overflow", ["--input-ids", input_path, "--max-tokens", 2147483648], "overflow"),
        ("no-tokenizer", ["--prompt", "not-supported"], "raw-ID"),
        ("no-vision", ["--image", "not-supported"], "vision"),
        ("teacher-policy", ["--input-ids", input_path, "--teacher-ids", input_path], "requires --fixed-steps"),
        ("teacher-short", ["--input-ids", input_path, "--teacher-ids", input_path,
                           "--fixed-steps", "--max-tokens", 4], "insufficient"),
    ]
    for compiler in COMPILERS:
        binary = ROOT / compiler / "qwen-raw"
        for name, args, diagnostic in bad_args:
            output = execute(binary, ["--model", models[64], *args], run / f"{compiler}-{name}.log", False)
            assert diagnostic.lower() in output.lower(), (name, output)
            all_counts[compiler]["rejects"] += 1
        for name, key, value in [("moe", "num_experts", 2), ("group", "hidden_size", 65),
                                 ("schedule", "full_attention_interval", 1)]:
            bad = copy.deepcopy(configs[64])
            bad["text_config"][key] = value
            write_json(config_path, bad)
            execute(binary, ["--model", models[64], "--input-ids", input_path],
                    run / f"{compiler}-config-{name}.log", False)
            all_counts[compiler]["rejects"] += 1
        write_json(config_path, configs[64])
        for mode, diagnostic in [("activation", "activation capacity"), ("capacity", "context capacity"),
                                 ("id", "out of vocabulary"), ("index-product", "index capacity"),
                                 ("int64-product", "64-bit multiplication overflow")]:
            output = execute(ROOT / compiler / "test_driver_api", [mode, models[64]],
                             run / f"{compiler}-api-{mode}.log", False)
            assert diagnostic in output, (mode, output)
            all_counts[compiler]["rejects"] += 1
        print(compiler + " " + json.dumps(all_counts[compiler], sort_keys=True), flush=True)
    write_json(run / "summary.json", all_counts)
    factor = 1 if sys.platform == "darwin" else 1024
    rss = {name: int(resource.getrusage(which).ru_maxrss * factor) for name, which in
           [("python_peak_bytes", resource.RUSAGE_SELF), ("child_peak_bytes", resource.RUSAGE_CHILDREN)]}
    assert sum(rss.values()) <= 64 * 1024 * 1024, rss
    write_json(run / "resident.json", rss)
    print("RESIDENT " + json.dumps(rss, sort_keys=True), flush=True)
    for case, *_ in cases:
        for owner in ("c", *COMPILERS):
            shutil.rmtree(run / case / owner)
    for compiler in COMPILERS:
        shutil.rmtree(run / f"{compiler}-padded-dump")


if __name__ == "__main__":
    main()
