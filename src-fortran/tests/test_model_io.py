"""Independent stdlib JSON/struct oracle. No MLX, NumPy, C, or model inference."""
import argparse
import copy
import json
import math
import pathlib
import shutil
import struct
import subprocess


ROOT = pathlib.Path("build/fortran-io")
FIXTURES = ROOT / "fixtures"


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_json(path, value):
    path.write_bytes(json_bytes(value))
    return path


def shard_file(path, header, payload=b""):
    raw = header if isinstance(header, bytes) else json_bytes(header)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


def tensor(dtype="BF16", shape=None, offsets=None):
    return {"dtype": dtype, "shape": [4] if shape is None else shape,
            "data_offsets": [0, 8] if offsets is None else offsets}


def config_fixture():
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5", "tie_word_embeddings": False,
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
        "eos_token_id": [127, 3],
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_act": "silu",
            "output_gate_type": "swish", "dtype": "bfloat16",
            "mamba_ssm_dtype": "float32",
            "attention_bias": False, "tie_word_embeddings": False, "attn_output_gate": True,
            "num_hidden_layers": 4, "hidden_size": 64, "intermediate_size": 128,
            "vocab_size": 128, "full_attention_interval": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4, "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6, "bos_token_id": 3, "pad_token_id": None,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default",
                                "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                                "mrope_section": [3, 3, 2]},
        },
    }


def write_archive(directory, tensors_by_shard, config=None):
    directory.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for i, entries in enumerate(tensors_by_shard, 1):
        filename = f"model-{i:05d}.safetensors"
        header, payload = {}, bytearray()
        for name, dtype, shape, raw in entries:
            start = len(payload)
            payload.extend(raw)
            header[name] = tensor(dtype, shape, [start, len(payload)])
            mapping[name] = filename
        shard_file(directory / filename, header, payload)
    write_json(directory / "model.safetensors.index.json", {"weight_map": mapping})
    if config is not None:
        write_json(directory / "config.json", config)
    return directory


def dense_fixture(directory):
    cfg = config_fixture()
    entries = []

    def add(name, dtype, shape):
        count = math.prod(shape)
        raw = bytes(count * {"BF16": 2, "U32": 4}[dtype])
        entries.append((name, dtype, shape, raw))

    def q8(name, k, n):
        add(name + ".weight", "U32", [n, k // 4])
        add(name + ".scales", "BF16", [n, k // 64])
        add(name + ".biases", "BF16", [n, k // 64])

    def vector(name, n):
        add(name, "BF16", [n])

    q8("language_model.model.embed_tokens", 64, 128)
    q8("language_model.lm_head", 64, 128)
    vector("language_model.model.norm.weight", 64)
    for li in range(4):
        p = f"language_model.model.layers.{li}"
        vector(p + ".input_layernorm.weight", 64)
        vector(p + ".post_attention_layernorm.weight", 64)
        if li != 3:
            for name, k, n in [("in_proj_qkv", 64, 128), ("in_proj_z", 64, 64),
                               ("in_proj_b", 64, 4), ("in_proj_a", 64, 4),
                               ("out_proj", 64, 64)]:
                q8(p + ".linear_attn." + name, k, n)
            add(p + ".linear_attn.conv1d.weight", "BF16", [128, 4, 1])
            vector(p + ".linear_attn.A_log", 4)
            vector(p + ".linear_attn.dt_bias", 4)
            vector(p + ".linear_attn.norm.weight", 16)
        else:
            for name, k, n in [("q_proj", 64, 256), ("k_proj", 64, 64),
                               ("v_proj", 64, 64), ("o_proj", 128, 64)]:
                q8(p + ".self_attn." + name, k, n)
            vector(p + ".self_attn.q_norm.weight", 64)
            vector(p + ".self_attn.k_norm.weight", 64)
        for name, k, n in [("gate_proj", 64, 128), ("up_proj", 64, 128), ("down_proj", 128, 64)]:
            q8(p + ".mlp." + name, k, n)
    return write_archive(directory, [entries[::2], entries[1::2]], cfg)


def metadata_oracle(directory):
    index = json.loads((directory / "model.safetensors.index.json").read_bytes())["weight_map"]
    descriptors, shards = {}, {}
    for filename in dict.fromkeys(index.values()):
        file = directory / filename
        with file.open("rb") as stream:
            length, = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
        shards[filename] = (file.stat().st_size, length)
        for name, desc in header.items():
            if name == "__metadata__":
                continue
            assert index[name] == filename
            assert name not in descriptors
            begin, end = desc["data_offsets"]
            descriptors[name] = (desc["dtype"], len(desc["shape"]), math.prod(desc["shape"]),
                                 end - begin, 8 + length + begin, *desc["shape"])
    assert set(descriptors) == set(index)
    return descriptors, shards, index


class Checks:
    def __init__(self, compiler):
        self.compiler = compiler
        self.binary = ROOT / compiler / "test_model_io"
        self.logs = ROOT / compiler / "checks"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def run(self, label, mode, path, *args, error=None):
        result = subprocess.run([str(self.binary), mode, str(path), *map(str, args)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        (self.logs / f"{label}.log").write_bytes(result.stdout + result.stderr)
        output = result.stdout.decode("utf-8")
        diagnostic = result.stderr.decode("utf-8", errors="replace")
        if error is None:
            assert result.returncode == 0, (self.compiler, label, output, diagnostic)
        else:
            assert result.returncode != 0 and "ERROR: " in diagnostic and error in diagnostic, (
                self.compiler, label, result.returncode, output, diagnostic)
        self.count += 1
        return output

    def metadata(self, label, directory, model=False, fixed_directory=False):
        mode = "archive-fixed" if fixed_directory else "model" if model else "archive"
        output = self.run(label, mode, directory)
        expected, shards, _ = metadata_oracle(directory)
        actual = {}
        shard_rows = []
        for line in output.splitlines():
            fields = line.split()
            if fields[0] == "T":
                actual[fields[1]] = (fields[2], *map(int, fields[3:]))
            elif fields[0] == "ARCHIVE":
                assert list(map(int, fields[1:])) == [len(shards), len(expected)]
            elif fields[0] == "SHARD":
                shard_rows.append(tuple(map(int, fields[2:])))
        assert actual == expected, (self.compiler, label, "metadata mismatch")
        assert shard_rows == list(shards.values())
        if model:
            config = json.loads((directory / "config.json").read_bytes())
            tc = config["text_config"]
            expected_config = [
                tc["num_hidden_layers"], tc["hidden_size"], tc["intermediate_size"], tc["vocab_size"],
                tc["layer_types"].count("linear_attention"), tc["layer_types"].count("full_attention"),
                tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"],
                tc["linear_num_key_heads"], tc["linear_num_value_heads"],
                tc["linear_key_head_dim"], tc["linear_value_head_dim"], tc["linear_conv_kernel_dim"],
                tc["num_attention_heads"] * tc["head_dim"], tc["num_key_value_heads"] * tc["head_dim"],
                tc["linear_num_key_heads"] * tc["linear_key_head_dim"],
                tc["linear_num_value_heads"] * tc["linear_value_head_dim"],
                2 * tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
                + tc["linear_num_value_heads"] * tc["linear_value_head_dim"],
                int(tc["head_dim"] * tc["rope_parameters"]["partial_rotary_factor"]),
                tc["max_position_embeddings"], 64, 8, tc["bos_token_id"], -1,
                tc.get("mtp_num_hidden_layers", 0),
            ]
            for line in output.splitlines():
                fields = line.split()
                if fields[0] == "CONFIG":
                    assert list(map(int, fields[1:])) == expected_config
                elif fields[0] == "PARAMS":
                    assert all(math.isclose(float(a), b, rel_tol=1e-7)
                               for a, b in zip(fields[1:], [tc["rms_norm_eps"],
                                   tc["rope_parameters"]["rope_theta"],
                                   tc["rope_parameters"]["partial_rotary_factor"]], strict=True))
                elif fields[0] == "EOS":
                    assert list(map(int, fields[1:])) == config["eos_token_id"]
                elif fields[0] == "MROPE":
                    rp = tc["rope_parameters"]
                    assert fields[1] == ("T" if rp["mrope_interleaved"] else "F")
                    assert list(map(int, fields[2:])) == rp["mrope_section"]
                elif fields[0] == "LAYERS":
                    assert fields[1:] == ["T" if x == "full_attention" else "F" for x in tc["layer_types"]]
            text = {k: v for k, v in expected.items() if k.startswith("language_model.")}
            assert f"TEXT {len(text)} {sum(v[3] for v in text.values())}" in output
        return expected

    def bits(self, label, directory, name, first, count):
        descriptors, _, mapping = metadata_oracle(directory)
        dtype, _, _, _, offset, *_ = descriptors[name]
        width, fmt = {"BF16": (2, "H"), "U32": (4, "I")}[dtype]
        with (directory / mapping[name]).open("rb") as stream:
            stream.seek(offset + (first - 1) * width)
            raw = stream.read(count * width)
        expected = struct.unpack("<" + fmt * count, raw)
        output = self.run(label, "bits", directory, name, first, count)
        fields = output.split()
        assert fields[0] == dtype
        assert tuple(int(x, 16) for x in fields[1:]) == expected


def run_fixtures(checks):
    c = checks
    c.metadata("dense-metadata", FIXTURES / "dense", model=True)
    c.metadata("tiny-metadata", FIXTURES / "tiny")
    c.metadata("fixed-length-directory", FIXTURES / "tiny", fixed_directory=True)
    for mode, value, expected in [
        ("ids", [3, 0, 127, 3], "IDS 3 0 127 3"),
        ("ids", {"other": True, "input_ids": [127, 0, 3, 3]}, "IDS 127 0 3 3"),
        ("ids", {"input_ids": [3], "input_ids ": [7]}, "IDS 3"),
    ]:
        path = write_json(FIXTURES / "ids.json", value)
        assert c.run(f"ids-{c.count}", mode, path, 128).strip() == expected
    for label, raw, message in [
        ("empty", b"[]", "must not be empty"),
        ("negative", b"[-1]", "outside vocabulary"),
        ("vocab", b"[128]", "outside vocabulary"),
        ("fraction", b"[1.0]", "expected integer"),
        ("exponent", b"[1e0]", "expected integer"),
        ("bool", b"[true]", "incorrect type"),
        ("nested", b"[[1]]", "incorrect type"),
        ("missing", b'{"a":[1]}', "missing required"),
        ("space-key", b'{"input_ids ":[1]}', "missing required"),
        ("wrong-root", b"1", "incorrect type"),
        ("overflow", b"[9223372036854775808]", "64-bit range"),
        ("leading-zero", b"[01]", "expected comma"),
        ("trailing-comma", b"[1,]", "expected JSON value"),
        ("truncated", b"[1,", "unexpected end"),
        ("trailing", b"[1]x", "trailing data"),
        ("duplicate", b'{"input_ids":[1],"input_\\u0069ds":[2]}', "duplicate object key"),
    ]:
        path = FIXTURES / "bad-ids.json"
        path.write_bytes(raw)
        c.run("ids-" + label, "ids", path, 128, error=message)
    path = write_json(FIXTURES / "ids.json", [2147483647, 0])
    assert c.run("ids-int32-max", "ids", path).strip() == "IDS 2147483647 0"
    write_json(path, [2147483648])
    c.run("ids-int32-overflow", "ids", path, error="int32 range")
    c.run("ids-bad-vocab", "ids", path, 0, error="vocabulary must")

    for number in [0, -9223372036854775808, 9223372036854775807]:
        path = write_json(FIXTURES / "number.json", number)
        assert c.run(f"integer-{number}", "integer", path).strip() == f"INTEGER {number}"
    for number in ["-9223372036854775809", "9223372036854775808"]:
        path.write_text(number)
        c.run("integer-overflow-" + number, "integer", path, error="64-bit range")

    path = FIXTURES / "strings.json"
    path.write_bytes(b'{"value":"a\\u0000\\b\\f\\n\\r\\t\\/\\\\\\"\\u00e9\\ud83d\\ude00"}')
    expected = 'a\0\b\f\n\r\t/\\\"é😀'.encode("utf-8").hex().upper()
    assert c.run("json-escapes", "string", path).strip() == "STRING " + expected
    write_json(path, {"value": "é中😀"})
    assert c.run("json-utf8", "string", path).strip() == "STRING " + "é中😀".encode().hex().upper()
    for label, raw, message in [
        ("escape", b'"\\q"', "invalid string escape"),
        ("high-surrogate", b'"\\ud800x"', "missing low surrogate"),
        ("low-surrogate", b'"\\udc00"', "unpaired low surrogate"),
        ("surrogate-pair", b'"\\ud800\\u0000"', "invalid low surrogate"),
        ("short-escape", b'"\\u0', "incomplete Unicode"),
        ("bad-hex", b'"\\uQQQQ"', "invalid Unicode"),
        ("overlong", b'"\xc0\x80"', "invalid UTF-8"),
        ("surrogate-utf8", b'"\xed\xa0\x80"', "invalid UTF-8"),
        ("high-utf8", b'"\xf4\x90\x80\x80"', "invalid UTF-8"),
        ("continuation", b'"\xe2ab"', "invalid UTF-8"),
        ("control", b'"\x00"', "unescaped control"),
        ("unterminated", b'"abc', "unterminated"),
        ("depth", b"[" * 66 + b"0" + b"]" * 66, "nesting exceeds"),
        ("number-dot", b"1.", "missing fractional"),
        ("number-exp", b"1e+", "missing exponent"),
        ("literal", b"trux", "invalid literal"),
    ]:
        path.write_bytes(raw)
        c.run("json-" + label, "json", path, error=message)

    bad = FIXTURES / "bad.safetensors"
    for label, header, payload, message in [
        ("dtype", {"x": tensor("F32")}, bytes(8), "unsupported tensor dtype"),
        ("dtype-space", {"x": tensor("BF16 ")}, bytes(8), "unsupported tensor dtype"),
        ("shape-type", {"x": tensor(shape="4")}, bytes(8), "incorrect type"),
        ("rank", {"x": tensor(shape=[1] * 7)}, bytes(8), "rank exceeds"),
        ("negative-dim", {"x": tensor(shape=[-4])}, bytes(8), "dimensions must be positive"),
        ("zero-dim", {"x": tensor(shape=[0])}, bytes(8), "dimensions must be positive"),
        ("fraction-dim", {"x": tensor(shape=[4.0])}, bytes(8), "expected integer"),
        ("product", {"x": tensor(shape=[9223372036854775807, 2])}, bytes(8), "multiplication overflow"),
        ("byte-overflow", {"x": tensor(shape=[9223372036854775807])}, bytes(8), "multiplication overflow"),
        ("negative-offset", {"x": tensor(offsets=[-1, 7])}, bytes(8), "outside shard"),
        ("reverse-offset", {"x": tensor(offsets=[8, 0])}, bytes(8), "outside shard"),
        ("offset-type", {"x": tensor(offsets=[0.0, 8])}, bytes(8), "expected integer"),
        ("offset-length", {"x": tensor(offsets=[0, 8, 9])}, bytes(8), "two integers"),
        ("bytecount", {"x": tensor(shape=[3])}, bytes(8), "bytecount mismatch"),
        ("data-truncated", {"x": tensor()}, bytes(7), "outside shard"),
        ("hole", {"x": tensor(offsets=[2, 10])}, bytes(10), "overlap or hole"),
        ("overlap", {"x": tensor(), "y": tensor()}, bytes(8), "overlap or hole"),
        ("extra-data", {"x": tensor()}, bytes(9), "trailing safetensors data"),
        ("metadata", {"x": tensor(), "__metadata__": {"format": 1}}, bytes(8), "incorrect type"),
        ("root-type", [], b"", "incorrect type"),
        ("missing-dtype", {"x": {"shape": [4], "data_offsets": [0, 8]}}, bytes(8), "missing required"),
        ("duplicate", b'{"x":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},'
                      b'"\\u0078":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}', bytes(8), "duplicate object key"),
    ]:
        shard_file(bad, header, payload)
        c.run("shard-" + label, "shard", bad, error=message)
    for label, raw, message in [
        ("short-prefix", b"\x01" * 7, "truncated safetensors length prefix"),
        ("short-header", struct.pack("<Q", 100) + b"{}", "truncated safetensors header"),
        ("huge-header", struct.pack("<Q", 2**31) + b"{}", "outside 2..16 MiB"),
        ("unsigned-header", struct.pack("<Q", 2**63) + b"{}", "exceeds signed 64-bit"),
        ("partial-json", struct.pack("<Q", 6) + b'{"x": ', "unexpected end"),
    ]:
        bad.write_bytes(raw)
        c.run("shard-" + label, "shard", bad, error=message)

    c.bits("tiny-u32", FIXTURES / "tiny", "words", 1, 4)
    c.bits("tiny-bf16", FIXTURES / "tiny", "halves", 1, 4)
    c.bits("tiny-subrange", FIXTURES / "tiny", "words", 2, 2)
    c.bits("name-exact", FIXTURES / "spaces", "x", 1, 1)
    c.bits("name-space", FIXTURES / "spaces", "x ", 1, 1)
    c.bits("chunk-u32", FIXTURES / "chunk", "words", 1, 16387)
    c.bits("chunk-bf16", FIXTURES / "chunk", "halves", 1, 32771)
    output = c.run("matrix-layout", "matrix", FIXTURES / "tiny")
    assert "MATRIX 00000000 800102FF FFFFFFFF 7FFFFFFF" in output
    assert "SCALE 0000 8000 7F81 FFFF" in output and "CUBE 1234 8001 ABCD FFFF" in output
    assert "BYTES 255 2 1 128" in output
    c.run("matrix-bad-shape", "matrix-bad-shape", FIXTURES / "tiny", error="dimensions mismatch")
    c.run("matrix-bad-rank", "matrix-bad-rank", FIXTURES / "tiny", error="rank mismatch")
    c.run("wrong-type", "wrong-type", FIXTURES / "tiny", "halves", error="dtype mismatch")
    c.run("bad-range", "bad-range", FIXTURES / "tiny", "words", error="first exceeds")
    c.run("range-end", "bits", FIXTURES / "tiny", "words", 4, 2, error="range exceeds")
    c.metadata("sparse-metadata", FIXTURES / "sparse")
    c.bits("sparse-tail", FIXTURES / "sparse", "tail", 1, 4)
    stale = write_archive(FIXTURES / ("stale-" + c.compiler),
                          [[("halves", "BF16", [4], bytes(8))]])
    c.run("stale-payload", "stale", stale, "halves", error="truncated tensor payload")
    stale = write_archive(FIXTURES / ("stale-tail-" + c.compiler),
                          [[("halves", "BF16", [4], bytes(8))]])
    c.run("stale-tail", "stale-tail", stale, "halves", error="truncated tensor payload")

    base = config_fixture()
    for label, section, key, value, message in [
        ("moe-architecture", None, "model_type", "qwen3_5_moe", "unsupported model_type"),
        ("type-space", None, "model_type", "qwen3_5 ", "unsupported model_type"),
        ("moe-field", "text_config", "num_experts", 256, "MoE is not supported"),
        ("missing-inter", "text_config", "intermediate_size", None, "missing required"),
        ("bad-integer", "text_config", "hidden_size", 64.0, "expected integer"),
        ("bad-boolean", "text_config", "attention_bias", 0, "incorrect type"),
        ("attention-bias", "text_config", "attention_bias", True, "unsupported attention_bias"),
        ("tied", "text_config", "tie_word_embeddings", True, "unsupported tie_word_embeddings"),
        ("ungated", "text_config", "attn_output_gate", False, "unsupported attn_output_gate"),
        ("state-dtype", "text_config", "mamba_ssm_dtype", "bfloat16", "unsupported mamba_ssm_dtype"),
        ("rope-scaling", "text_config", "rope_scaling", {"type": "linear", "factor": 2}, "unsupported rope_scaling"),
        ("bad-heads", "text_config", "num_key_value_heads", 3, "incompatible head"),
        ("dimension-overflow", "text_config", "num_attention_heads", 2147483647, "derived dimension"),
        ("no-hybrid", "text_config", "full_attention_interval", 1, "hybrid"),
        ("eps-negative", "text_config", "rms_norm_eps", -1, "finite, positive"),
        ("eps-overflow", "text_config", "rms_norm_eps", 1e300, "representable as real32"),
        ("layer-count", "text_config", "layer_types", ["full_attention"], "cover every layer"),
        ("layer-order", "text_config", "layer_types", ["full_attention"] * 4, "interval mismatch"),
        ("bits", "quantization", "bits", 4, "only 8-bit"),
        ("groups", "quantization", "group_size", 32, "group_size=64"),
        ("mode", "quantization", "mode", "mxfp4", "unsupported mode"),
        ("eos", None, "eos_token_id", [128], "outside vocabulary"),
    ]:
        cfg = copy.deepcopy(base)
        obj = cfg if section is None else cfg[section]
        if value is None:
            del obj[key]
        else:
            obj[key] = value
        path = write_json(FIXTURES / "bad-config.json", cfg)
        c.run("config-" + label, "config", path, error=message)
    cfg = copy.deepcopy(base)
    cfg["text_config"]["rope_parameters"]["rope_type"] = "yarn"
    c.run("config-rope", "config", write_json(path, cfg), error="unsupported rope_type")
    cfg = copy.deepcopy(base)
    cfg["quantization_config"] = {"bits": 4, "group_size": 64, "mode": "affine"}
    c.run("config-quant-conflict", "config", write_json(path, cfg), error="only 8-bit")

    # Generic storage accepts valid names/shapes that model semantics must reject.
    for label, change, message in [
        ("conv-layout", "conv", "incompatible dimensions"),
        ("normalization", "mtp", "unsanitized normalization"),
        ("extra-text", "extra", "unexpected text tensor"),
        ("missing-text", "missing", "missing tensor"),
    ]:
        directory = FIXTURES / ("semantic-" + label)
        dense_fixture(directory)
        idx = json.loads((directory / "model.safetensors.index.json").read_bytes())
        target = "language_model.model.layers.0.linear_attn.conv1d.weight"
        if change == "conv":
            file = directory / idx["weight_map"][target]
            data = file.read_bytes()
            length, = struct.unpack("<Q", data[:8])
            header = json.loads(data[8:8 + length])
            header[target]["shape"] = [128, 1, 4]
            shard_file(file, header, data[8 + length:])
        else:
            mapping = idx["weight_map"]
            old = "language_model.model.norm.weight"
            new = {"mtp": "mtp.norm.weight", "extra": "language_model.extra.weight",
                   "missing": "vision.norm.weight"}[change]
            file = directory / mapping[old]
            data = file.read_bytes()
            length, = struct.unpack("<Q", data[:8])
            header = json.loads(data[8:8 + length])
            if change == "extra":
                # Rename a vision entry is unnecessary: append a genuine additional text scalar.
                header[new] = tensor("BF16", [], [len(data) - 8 - length, len(data) - 8 - length + 2])
                mapping[new] = mapping[old]
                shard_file(file, header, data[8 + length:] + bytes(2))
            else:
                header[new] = header.pop(old)
                mapping[new] = mapping.pop(old)
                shard_file(file, header, data[8 + length:])
            write_json(directory / "model.safetensors.index.json", idx)
        c.run("weights-" + label, "model", directory, error=message)

    for label, mapping, headers, message in [
        ("wrong-shard", {"x": "b.safetensors", "z": "a.safetensors"},
         {"a.safetensors": {"x": tensor()}, "b.safetensors": {"z": tensor()}}, "shard mismatch"),
        ("missing", {"x": "a.safetensors", "z": "a.safetensors"},
         {"a.safetensors": {"x": tensor()}}, "references missing tensor"),
        ("unindexed", {"z": "a.safetensors"}, {"a.safetensors": {"x": tensor()}}, "absent from weight_map"),
        ("duplicate", {"x": "a.safetensors", "z": "b.safetensors"},
         {"a.safetensors": {"x": tensor()}, "b.safetensors": {
             "x": tensor(), "z": tensor(offsets=[8, 16])}}, "duplicate tensor across shards"),
        ("path", {"x": "../a.safetensors"}, {}, "local .safetensors basename"),
    ]:
        directory = FIXTURES / ("mapping-" + label)
        directory.mkdir(exist_ok=True)
        for filename, header in headers.items():
            shard_file(directory / filename, header, bytes(8 * len(header)))
        write_json(directory / "model.safetensors.index.json", {"weight_map": mapping})
        c.run("mapping-" + label, "archive", directory, error=message)
    directory = FIXTURES / "mapping-size"
    write_archive(directory, [[("x", "BF16", [4], bytes(8))]])
    index = json.loads((directory / "model.safetensors.index.json").read_bytes())
    index["metadata"] = {"total_size": 9}
    write_json(directory / "model.safetensors.index.json", index)
    c.run("mapping-size", "archive", directory, error="total_size disagrees")
    index["metadata"]["total_size"] = 8.0
    write_json(directory / "model.safetensors.index.json", index)
    c.run("mapping-size-type", "archive", directory, error="expected integer")
    index["metadata"]["total_size"] = 8
    write_json(directory / "model.safetensors.index.json", index)
    c.metadata("mapping-size-valid", directory)


def make_fixtures():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    dense_fixture(FIXTURES / "dense")
    write_archive(FIXTURES / "tiny", [[
        ("words", "U32", [2, 2], struct.pack("<4I", 0, 0x800102FF, 0xFFFFFFFF, 0x7FFFFFFF)),
        ("halves", "BF16", [2, 2], struct.pack("<4H", 0, 0x8000, 0x7F81, 0xFFFF)),
        ("cube", "BF16", [1, 2, 2], struct.pack("<4H", 0x1234, 0x8001, 0xABCD, 0xFFFF)),
    ]])
    write_archive(FIXTURES / "chunk", [[
        ("words", "U32", [16387], struct.pack("<16387I",
            *((i * 2654435761) & 0xFFFFFFFF for i in range(16387)))),
        ("halves", "BF16", [32771], struct.pack("<32771H",
            *((i * 40503) & 0xFFFF for i in range(32771)))),
    ]])
    write_archive(FIXTURES / "spaces", [[
        ("x", "BF16", [1], struct.pack("<H", 0x3F80)),
        ("x ", "BF16", [1], struct.pack("<H", 0x8001)),
    ]])
    sparse = FIXTURES / "sparse"
    sparse.mkdir(exist_ok=True)
    padding = 2**31 + 68
    header = json_bytes({
        "padding": tensor("U32", [padding // 4], [0, padding]),
        "tail": tensor("BF16", [4], [padding, padding + 8]),
    })
    file = sparse / "model.safetensors"
    with file.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)) + header)
        stream.seek(8 + len(header) + padding)
        stream.write(struct.pack("<4H", 0x3F80, 0x8000, 0xFFFF, 0xABCD))
    assert file.stat().st_size > 2**31 and file.stat().st_blocks * 512 < 1024**2
    write_json(sparse / "model.safetensors.index.json",
               {"weight_map": {"padding": file.name, "tail": file.name}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path)
    args = parser.parse_args()
    make_fixtures()
    for compiler in ("gfortran", "flang"):
        checks = Checks(compiler)
        run_fixtures(checks)
        if args.model is not None:
            descriptors = checks.metadata("actual-model-metadata", args.model, model=True)
            text = {k: v for k, v in descriptors.items() if k.startswith("language_model.")}
            assert len(descriptors) == 2180 and len(text) == 1847
            assert sum(v[3] for v in text.values()) == 28579478528
            for label, name, count in [
                ("actual-norm-bf16", "language_model.model.norm.weight", 5120),
                ("actual-gate-bf16", "language_model.model.layers.0.linear_attn.A_log", 48),
                ("actual-q8-u32", "language_model.model.layers.0.linear_attn.in_proj_a.weight", 16),
            ]:
                checks.bits(label, args.model, name, 1, count)
            print(f"{compiler}: actual metadata 2180/1847, text bytes 28579478528; "
                  "selected payload 10336 BF16 bytes + 64 U32 bytes match Python struct")
        print(f"{compiler}: {checks.count} focused checks passed; logs: {checks.logs}")
    # Keep logs, not generated inputs. Failed runs retain fixtures for diagnosis.
    shutil.rmtree(FIXTURES)


if __name__ == "__main__":
    main()
