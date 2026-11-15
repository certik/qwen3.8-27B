"""CPU-only native-semantic tests. Historical scalar-C fixtures are never rewritten."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import time

from test_driver import fixture, tensor, write_json
from test_model_io import write_archive

OUT = Path("artifacts/fortran-native")
BUILD = Path("build/fortran-driver")
REPORT = []


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf(x):
    w = struct.unpack("<I", struct.pack("<f", x))[0]
    return struct.unpack("<f", struct.pack("<I", ((w + 0x7fff + ((w >> 16) & 1)) >> 16) << 16))[0]


def bits(values):
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", bf(v)))[0] >> 16) for v in values)


def floats(raw, fp32=False):
    if fp32:
        result = array.array("f")
        result.frombytes(raw)
        return result
    words = array.array("H")
    words.frombytes(raw)
    return [struct.unpack("<f", struct.pack("<I", w << 16))[0] for w in words]


def compare(actual, expected, label, atol=0.0, rtol=0.0, fp32=False):
    assert len(actual) == len(expected), (label, len(actual), len(expected))
    a, e = floats(actual, fp32), floats(expected, fp32)
    assert all(math.isfinite(x) for x in a) and all(math.isfinite(x) for x in e), label
    errors = [abs(x-y) for x, y in zip(a, e)]
    bad = [i for i, (x, y) in enumerate(zip(a, e)) if abs(x-y) > atol+rtol*abs(y)]
    receipt = dict(label=label, elements=len(a), byte_exact=actual == expected,
                   different_values=sum(x != y for x, y in zip(a, e)),
                   maximum_absolute=max(errors, default=0), outside_bound=len(bad),
                   atol=atol, rtol=rtol, candidate_sha256=hashlib.sha256(actual).hexdigest(),
                   reference_sha256=hashlib.sha256(expected).hexdigest())
    REPORT.append(receipt)
    assert not bad, (receipt, bad[:8])


def execute(compiler, mode, source, name, dimensions):
    output = OUT / compiler / (name + ".bin")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(BUILD / compiler / "native-replay"), mode, str(source), str(output), *map(str, dimensions)]
    with output.with_suffix(".log").open("wb") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert process.returncode == 0, (command, output.with_suffix(".log").read_text())
    return output.read_bytes(), output


def sigmoid(x):
    e = bf(math.exp(abs(x)))
    y = bf(1.0/bf(1.0+e))
    return y if x < 0 else bf(1.0-y)


def fixtures(compiler):
    directory = OUT / "typed-fixtures"
    directory.mkdir(exist_ok=True)
    # Negative/positive affine biases, high-bit packed words and BF16 midpoints.
    rng = random.Random(861)
    for m, k, n in [(1,512,8), (1,512,9), (18,512,8), (2,64,9), (3,192,11)]:
        name = f"linear-{m}-{k}-{n}"
        case = directory / name
        case.mkdir(exist_ok=True)
        x = [bf(rng.uniform(-2, 2)) for _ in range(m*k)]
        codes = [rng.randrange(256) for _ in range(n*k)]
        scales = [bf(rng.uniform(.001, .02)) for _ in range(n*k//64)]
        biases = [bf(rng.uniform(-1, 1)) for _ in scales]
        for file, raw in [("x.bin", bits(x)), ("w.bin", bytes(codes)),
                          ("s.bin", bits(scales)), ("b.bin", bits(biases))]:
            (case/file).write_bytes(raw)
        vector = m < (18 if k <= 2048 and n <= 2048 else 12 if k <= 4096 and n <= 4096 else 10)
        vector = vector and k % 512 == 0 and n % 8 == 0
        parts = min(max(1, 512//(((m+31)//32)*((n+31)//32))), k//64)
        while (k//64) % parts:
            parts -= 1
        expected = []
        for token in range(m):
            for row in range(n):
                terms = []
                for group in range(k//64):
                    sb = row*k//64+group
                    xx = x[token*k+group*64:token*k+(group+1)*64]
                    cc = codes[row*k+group*64:row*k+(group+1)*64]
                    if vector:
                        dot = f32(sum(f32(c*v) for c, v in zip(cc, xx)))
                        sx = f32(sum(xx))
                        terms.append(f32(f32(scales[sb]*dot)+f32(biases[sb]*sx)))
                    else:
                        terms.extend(f32(bf(f32(c*scales[sb]+biases[sb]))*v) for c, v in zip(cc, xx))
                if vector:
                    value = bf(f32(sum(terms)))
                elif parts == 1:
                    value = bf(f32(sum(terms)))
                else:
                    width = k//parts
                    partials = [bf(f32(sum(terms[p*width:(p+1)*width]))) for p in range(parts)]
                    while len(partials) > 1:
                        pairs = [bf(partials[p]+partials[p+1]) for p in range(0,len(partials)-1,2)]
                        if len(partials) % 2:
                            pairs.append(partials[-1])
                        partials = pairs
                    value = partials[0]
                expected.append(value)
        actual, _ = execute(compiler, "linear", case, name, [m,k,n])
        # SUM association is compiler-owned, not scalar-C/Kahan/Apple lanes.
        compare(actual, bits(expected), compiler+"/"+name, .03125, .0078125)

    x = list(map(bf, [-11.5, -3.015625, -.50390625, -.00390625, 0, .00390625, .50390625, 3.015625, 11.5]))
    up = list(map(bf, [.5, -.9375, 1.00390625, -2.015625, 2, .1279296875, -.50390625, 1.5, -1]))
    (directory/"x.bin").write_bytes(bits(x))
    (directory/"b.bin").write_bytes(bits(up))
    for mode in ["sigmoid", "silu", "swiglu", "gate", "gated", "add"]:
        expected = []
        for g, u in zip(x, up):
            s = sigmoid(g)
            stable = f32(1/f32(1+f32(math.exp(abs(g)))))
            if g >= 0:
                stable = f32(1-stable)
            expected.append(dict(sigmoid=s, silu=bf(g*s), swiglu=bf(bf(g*s)*u),
                                 gate=bf(s*u), gated=bf(f32(f32(g*stable)*u)), add=bf(g+u))[mode])
        actual, _ = execute(compiler, mode, directory, mode, [len(x),1,1])
        compare(actual, bits(expected), compiler+"/"+mode)
    alog = list(map(bf, [-2, -.5, .25, 0, 1, -.25, .5, -1, 2]))
    bias = list(map(bf, [.5, -.25, .00390625, 0, -.5, .015625, .25, -.125, 1]))
    (directory/"a_log.bin").write_bytes(bits(alog))
    (directory/"dt_bias.bin").write_bytes(bits(bias))
    expected = []
    for a, al, b in zip(x, alog, bias):
        added = bf(a+b)
        e = bf(math.exp(-abs(added)))
        sp = bf(max(added,0)+bf(math.log1p(e)))
        expected.append(f32(math.exp(f32(-f32(math.exp(al))*sp))))
    actual, _ = execute(compiler, "g", directory, "multiplicative-g", [len(x),1,1])
    compare(actual, struct.pack("<"+"f"*len(x), *expected), compiler+"/multiplicative-g",
            1e-7, 1e-6, fp32=True)
    # Rounded subtraction creates a first-index tie unlike raw logits argmax.
    logits = [0.99609375, 1.0]+[0.0]*100
    (directory/"x.bin").write_bytes(bits(logits))
    logits = list(map(bf, logits))
    lse = bf(max(logits)+math.log(sum(math.exp(v-max(logits)) for v in logits)))
    expected = [bf(v-lse) for v in logits]
    actual, path = execute(compiler, "sampler", directory, "sampler", [len(logits),1,1])
    compare(actual, bits(expected), compiler+"/sampler-logprobs")
    assert "id=0" in path.with_suffix(".log").read_text()
    assert logits.index(max(logits)) == 1
    # Nondefault epsilon and normalize-before-weight stores.
    xx = list(map(bf, [.501953125, -.99609375, 1.0078125, -2.015625, .125, -3.015625, .00390625]))
    ww = list(map(bf, [1.0078125, -.75, 1.5, .99609375, .25, -.5, 2]))
    (directory/"x.bin").write_bytes(bits(xx))
    (directory/"weight.bin").write_bytes(bits(ww))
    eps = f32(.007)
    rr = f32(1/math.sqrt(f32(f32(sum(v*v for v in xx))/len(xx)+eps)))
    for mode, weighted in [("rms", 0), ("rms", 1), ("scaled-rms", 0)]:
        args = [1,len(xx),weighted,eps]
        expected = [bf(f32(v*rr)) for v in xx]
        if weighted:
            expected = [bf(v*w) for v,w in zip(expected,ww)]
        if mode == "scaled-rms":
            args.append(.08838834764831845)
            expected = [bf(v*bf(args[-1])) for v in expected]
        actual, _ = execute(compiler, mode, directory, mode+str(weighted), args)
        compare(actual, bits(expected), compiler+"/"+mode+str(weighted))
    for length in [1,2,3,5]:
        channels, kernel = 7, 4
        xx = [bf(rng.uniform(-.7,.7)) for _ in range(channels*length)]
        carry = [bf(rng.uniform(-.7,.7)) for _ in range(channels*(kernel-1))]
        weight = [bf(rng.uniform(-.5,.5)) for _ in range(channels*kernel)]
        (directory/"x.bin").write_bytes(bits(xx))
        (directory/"carry.bin").write_bytes(bits(carry))
        (directory/"weight.bin").write_bytes(bits(weight))
        joined = carry+xx
        expected = []
        for t in range(length):
            for c in range(channels):
                expected.append(bf(f32(sum(f32(joined[(t+j)*channels+c]*weight[c*kernel+j])
                                           for j in range(kernel)))))
        actual, path = execute(compiler, "conv", directory, f"conv-{length}", [length,channels,kernel])
        compare(actual, bits(expected), compiler+f"/conv-{length}")
        compare(Path(str(path)+".carry").read_bytes(), bits(joined[-channels*(kernel-1):]),
                compiler+f"/conv-carry-{length}")


def saved_evidence(compiler, native):
    # Reuse complete same-input GPU receipts, without opening a device/library.
    pinned = Path("/Users/ondrej/.copilot/session-state/9e308d73-081d-4371-a886-3e7ab8cb33da/files/mlx-exact-runtime/pilot")
    qmv_receipts = json.loads((pinned/"results.json").read_text())["cases"]
    for name in ["k5120-n48", "k5120-n6144-varied", "k6144-n5120"]:
        row = next(r for r in qmv_receipts if r["case"] == name)
        source = pinned/"cases"/name
        for key, sha in row["input_sha256"].items():
            assert hashlib.sha256((source/(key+".bin")).read_bytes()).hexdigest() == sha
        native_output, c_output = (source/"native.bin").read_bytes(), (source/"frozen-c.bin").read_bytes()
        assert hashlib.sha256(native_output).hexdigest() == row["native_output_sha256"]
        assert hashlib.sha256(c_output).hexdigest() == row["frozen_C_output_sha256"]
        ranges = {}
        for filename in ["x.bin","s.bin","b.bin"]:
            typed = floats((source/filename).read_bytes())
            assert all(math.isfinite(v) for v in typed)
            ranges[filename] = dict(min=min(typed),max=max(typed),distinct=len(set(typed)))
        assert ranges["x.bin"]["min"] < 0 < ranges["x.bin"]["max"]
        assert ranges["s.bin"]["min"] > 0 and ranges["b.bin"]["min"] < 0
        assert len(set(floats(native_output))) > 1
        REPORT.append(dict(label=compiler+"/typed-audit/"+name,ranges=ranges,immutable_input_hashes=True))
        actual, path = execute(compiler,"linear",source,"qmv-"+name,[row["M"],row["K"],row["N"]])
        assert "qmv=T" in path.with_suffix(".log").read_text()
        compare(actual,native_output,compiler+"/saved-native/qmv/"+name,.03,.03)
        compare(actual,c_output,compiler+"/fixed-C/qmv/"+name,row["existing_linear_atol"],row["existing_linear_rtol"])
    math_root = native/"artifacts/native-math/guarded-operator-checks-02"
    proof = json.loads((math_root/"proof.json").read_text())
    for row in proof["checks"]:
        if row["family"] == "embedding" and row["index_dtype"] == "int32":
            name = f"embedding-{row['rows']}-i32"
            source = math_root/name
            for filename, sha in row["saved_sha256"].items():
                assert hashlib.sha256((source/(filename+".bin")).read_bytes()).hexdigest() == sha
            actual, _ = execute(compiler, "embedding", source, name, [row["rows"],5120,64])
            compare(actual,(source/"native.bin").read_bytes(),compiler+"/saved-native/"+name)
            continue
        if row["family"] not in ("attention", "rope"):
            continue
        m = row["rows"]
        if m > 17:
            continue  # No 2K CPU all-pairs work for initial primitive diagnosis.
        if row["family"] == "attention":
            name = f"attention-{m}-{row['length']}"
            dimensions = [m,256,24,row["length"],row["capacity"],0]
        else:
            name = f"rope-{m}-{row['offset']}-{'inplace' if row['inplace'] else 'out'}"
            dimensions = [m,256,24,row["offset"]]
        source = math_root/name
        for filename, sha in row["saved_sha256"].items():
            assert hashlib.sha256((source/(filename+".bin")).read_bytes()).hexdigest() == sha
        actual, _ = execute(compiler, row["family"], source, name, dimensions)
        compare(actual, (source/"native.bin").read_bytes(), compiler+"/saved-native/"+name, .05, .05)
    for m, k, n in [(17,5120,48), (17,5120,1024)]:
        name = f"m{m}-k{k}-n{n}"
        source = native/"artifacts/native-qmm/pilot"/name
        identity = json.loads((source/"native-result.json").read_text())
        for key, sha in identity["inputs_sha256"].items():
            assert hashlib.sha256((source/(key+".bin")).read_bytes()).hexdigest() == sha
        actual, _ = execute(compiler, "linear", source, name, [m,k,n])
        expected = (source/"native-y.bf16").read_bytes()
        assert hashlib.sha256(expected).hexdigest() == identity["native_y_sha256"]
        compare(actual, expected, compiler+"/saved-native/qmm/"+name, .05, .05)
        # Historical fixed C tolerance remains a separate receipt and gate.
        compare(actual, (source/"frozen-C-y.bf16").read_bytes(), compiler+"/fixed-C/qmm/"+name, .03, .03)
    for m in [1,17]:
        source = native/f"artifacts/native-recurrent/pilot/T{m}"
        actual, path = execute(compiler, "recurrent", source, f"recurrent-T{m}", [m,128,48])
        compare(actual, (source/"native-y.bf16").read_bytes(), compiler+f"/saved-native/recurrent-T{m}", .05, .05)
        compare(actual, (source/"frozen-C-y.bf16").read_bytes(), compiler+f"/fixed-C/recurrent-T{m}", .05, .05)
        state = Path(str(path)+".state").read_bytes()
        compare(state, (source/"native-state.f32").read_bytes(), compiler+f"/native-state-T{m}", 1e-5, 1e-5, True)
        compare(state, (source/"frozen-C-state.f32").read_bytes(), compiler+f"/fixed-C-state-T{m}", 1e-5, 1e-5, True)


def attention_boundaries(compiler):
    rng = random.Random(641)
    for rows, length, offset in [(5,7,2), (6,7,1), (1,1023,1022), (1,1024,1023), (1,1025,1024)]:
        source = OUT/f"attention-fixture-{rows}-{length}"
        source.mkdir(exist_ok=True)
        width, heads = 256, 6
        q = [bf(rng.uniform(-.3,.3)) for _ in range(heads*rows*width)]
        k = [bf(rng.uniform(-.3,.3)) for _ in range(length*width)]
        v = [bf(rng.uniform(-.7,.7)) for _ in range(length*width)]
        for name, values in [("q",q),("k",k),("v",v)]:
            (source/(name+".bin")).write_bytes(bits(values))
        expected = []
        generic = rows*6 > 32
        blocks = 0 if generic else 1 if length < 1024 else 64 if length == 1024 else 128
        for h in range(heads):
            for token in range(rows):
                visible = min(length,offset+token+1) if rows > 1 else length
                query = [f32(x*.0625) for x in q[(h*rows+token)*width:(h*rows+token+1)*width]]
                if generic:
                    query = list(map(bf, query))
                scores = [f32(sum(f32(query[d]*k[t*width+d]) for d in range(width))) for t in range(visible)]
                if generic:
                    scores = list(map(bf, scores))
                if blocks > 1:
                    maxima, sums, partials = [], [], []
                    for b in range(blocks):
                        maximum, total, partial = -3.4028234663852886e38, 0.0, [0.0]*width
                        for t in range(b,visible,blocks):
                            new_max = max(maximum,scores[t])
                            factor, e = f32(math.exp(f32(maximum-new_max))), f32(math.exp(f32(scores[t]-new_max)))
                            total = f32(f32(total*factor)+e)
                            partial = [f32(f32(x*factor)+f32(e*v[t*width+d])) for d,x in enumerate(partial)]
                            maximum = new_max
                        maxima.append(maximum)
                        sums.append(total)
                        partials.append(list(map(bf, partial)))
                    maximum = max(maxima)
                    factors = [f32(math.exp(f32(x-maximum))) for x in maxima]
                    total = f32(sum(f32(x*y) for x,y in zip(sums,factors)))
                    expected.extend(bf(f32(f32(sum(f32(partials[b][d]*factors[b]) for b in range(blocks)))/total))
                                    for d in range(width))
                else:
                    maximum = max(scores)
                    weights = [f32(math.exp(f32(x-maximum))) for x in scores]
                    total = f32(sum(weights))
                    if generic:
                        weights = [bf(f32(x/total)) for x in weights]
                    for d in range(width):
                        output = f32(sum(f32(weights[t]*v[t*width+d]) for t in range(visible)))
                        if not generic:
                            output = f32(output/total)
                        expected.append(bf(output))
        actual, path = execute(compiler,"attention",source,f"attention-boundary-{rows}-{length}",
                               [rows,width,heads,length,length,offset])
        compare(actual,bits(expected),compiler+f"/attention-boundary-{rows}-{length}",.0005,.015625)
        assert f"attention-blocks={blocks}" in path.with_suffix(".log").read_text()


def tiny_graph(compilers):
    directory = OUT/"tiny-model"
    directory.mkdir(exist_ok=True)
    cfg = fixture(directory,256)
    cfg["text_config"]["max_position_embeddings"] = 4096
    write_json(directory/"config.json",cfg)
    reports = {}
    for compiler in compilers:
        for prompt, generated in [(1,1), (5,3), (6,2), (18,17), (18,65), (1023,1), (2049,1), (2050,1)]:
            name = f"tiny-P{prompt}-G{generated}"
            source = OUT/(name+"-input.json")
            write_json(source, [1+i%126 for i in range(prompt)])
            result = OUT/compiler/(name+".json")
            result.unlink(missing_ok=True)
            command = [str(BUILD/compiler/"qwen-raw"), "--model", str(directory), "--input-ids", str(source),
                       "--max-tokens", str(generated), "--fixed-steps", "--reset-check", "--result", str(result)]
            with result.with_suffix(".log").open("wb") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            assert process.returncode == 0, result.with_suffix(".log").read_text()
            receipt = json.loads(result.read_text())
            assert receipt["schema"] == 3 and receipt["actual_generated"] == generated
            prefixes = (prompt-2)//2048+1 if prompt > 1 else 0
            assert receipt["cache_only_forward_calls"] == prefixes
            assert receipt["terminal_prefixes_pruned"] == prefixes
            assert receipt["head_calls"] == receipt["sampler_calls"] == generated+1
            assert receipt["cached_step_count"] == generated and receipt["emitted_cached_count"] == generated-1
            assert receipt["inference_forward_calls"] == generated+1+prefixes
            assert receipt["reset_forward_calls"] == receipt["inference_forward_calls"]
            assert receipt["inference_cache_tokens_processed"] == prompt+generated
            assert receipt["cache_tokens_processed"] == 2*(prompt+generated)
            assert receipt["extra_unemitted_decode_forwards"] == 1 and receipt["reset_cache_and_logits_exact"]
            assert receipt["recurrent_step_calls"] == 3*(prompt+generated)
            vector_heads = max(0,min(generated+1,1024-prompt))
            assert receipt["attention_vector_calls"] == vector_heads
            assert receipt["attention_two_pass_calls"] == generated+1-vector_heads
            assert receipt["attention_generic_calls"] == 0
            assert receipt["quantized_matrix_calls"] > 0 and receipt["quantized_vector_calls"] == 0
            assert len(receipt["consumed_decode_ids"]) == generated
            assert receipt["consumed_decode_ids"] == receipt["generated_ids"]
            assert receipt["throughput_valid"] is False
            reports[compiler,name] = receipt
            REPORT.append(dict(label=compiler+"/"+name, protocol="PASS", complete_reset="PASS",
                               native_full_model_acceptance=False))
    if len(compilers) == 2:
        for name in {name for _,name in reports}:
            for field in ["generated_ids", "consumed_decode_ids", "lookahead_id"]:
                assert reports[compilers[0],name][field] == reports[compilers[1],name][field], (name,field)
    for compiler in compilers:
        source = OUT/"edge-input.json"
        write_json(source,[1,2,3,4,5])
        teacher = OUT/"edge-teacher.json"
        write_json(teacher,[127,0,11])
        result = OUT/compiler/"teacher.json"
        base = [str(BUILD/compiler/"qwen-raw"),"--model",str(directory),"--input-ids",str(source)]
        command = [*base,"--max-tokens","3","--fixed-steps","--reset-check","--teacher-ids",str(teacher),
                   "--result",str(result)]
        with result.with_suffix(".log").open("wb") as log:
            process = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        assert process.returncode == 0, result.with_suffix(".log").read_text()
        actual = json.loads(result.read_text())
        assert actual["consumed_decode_ids"] == [127,0,11] and actual["teacher_forced"] is True
        assert actual["reset_cache_and_logits_exact"] is True and actual["head_calls"] == 4
        write_json(teacher,[127,0])
        with (OUT/compiler/"short-teacher.log").open("wb") as log:
            process = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=30)
        assert process.returncode != 0
        assert "insufficient teacher continuation IDs" in (OUT/compiler/"short-teacher.log").read_text()
        config = directory/"config.json"
        original = json.loads(config.read_text())
        changed = json.loads(config.read_text())
        changed["eos_token_id"] = reports[compiler,"tiny-P5-G3"]["generated_ids"][0]
        write_json(config,changed)
        try:
            result = OUT/compiler/"early-eos.json"
            with result.with_suffix(".log").open("wb") as log:
                process = subprocess.run([*base,"--max-tokens","3","--reset-check","--result",str(result)],
                                         stdout=log,stderr=subprocess.STDOUT,timeout=180)
            assert process.returncode == 0, result.with_suffix(".log").read_text()
            actual = json.loads(result.read_text())
            assert actual["actual_generated"] == 1 and len(actual["consumed_decode_ids"]) == 1
            assert actual["head_calls"] == actual["sampler_calls"] == 2
            assert actual["inference_forward_calls"] == 3 and actual["reset_forward_calls"] == 3
            assert actual["inference_cache_tokens_processed"] == 6
            assert actual["reset_cache_and_logits_exact"] is True
        finally:
            write_json(config,original)
        with (OUT/compiler/"state-bits.log").open("wb") as log:
            process = subprocess.run([str(BUILD/compiler/"test_driver_api"),"state-bits"],
                                     stdout=log,stderr=subprocess.STDOUT,timeout=30)
        assert process.returncode == 0
        REPORT.append(dict(label=compiler+"/request-edges",teacher_includes_lookahead="PASS",
                           short_teacher_rejection="PASS",EOS_lookahead_and_reset="PASS",exact_state_bits="PASS"))


def terminal_recurrent_prefix(compilers):
    seed = OUT/"terminal-prefix-seed"
    seed.mkdir()
    cfg = fixture(seed,256)
    entries = []
    for shard in sorted(seed.glob("*.safetensors")):
        raw = shard.read_bytes()
        header_bytes, = struct.unpack_from("<Q",raw)
        header = json.loads(raw[8:8+header_bytes])
        payload = raw[8+header_bytes:]
        for name, info in header.items():
            start, end = info["data_offsets"]
            entries.append((name,info["dtype"],info["shape"],payload[start:end]))
    entries += [(name.replace(".layers.2.",".layers.4."),dtype,shape,raw)
                for name,dtype,shape,raw in entries if ".layers.2." in name]
    cfg["text_config"]["num_hidden_layers"] = 5
    cfg["text_config"]["layer_types"].append("linear_attention")
    model = OUT/"terminal-recurrent-model"
    write_archive(model,[entries[::2],entries[1::2]],cfg)
    for compiler in compilers:
        dump = OUT/compiler/"terminal-prefix-dump"
        dump.mkdir(parents=True)
        command = [str(BUILD/compiler/"test_driver_api"),"terminal-recurrent-prefix",str(model),str(dump)]
        log_path = OUT/compiler/"terminal-prefix.log"
        with log_path.open("wb") as log:
            process = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        assert process.returncode == 0, log_path.read_text()
        assert not (dump/"z_4.bin").exists(), "pruned terminal z projection was incorrectly dumped"
        assert (dump/"qkv_4.bin").exists() and (dump/"conv_out_4.bin").exists()
        shape, payload = tensor(dump/"delta_state_4.bin")
        values = floats(payload,fp32=True)
        assert shape == (3,128,128) and all(math.isfinite(v) for v in values)
        assert max(map(abs,values)) > 0
        REPORT.append(dict(label=compiler+"/terminal-recurrent-prefix",
                           pruned_z_absent=True,complete_cache_state_matches_full_forward=True,
                           required_state_dump_finite=True,real_model_acceptance=False))


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", nargs="+", choices=["flang","gfortran"], default=["flang","gfortran"])
    parser.add_argument("--native-evidence", type=Path)
    parser.add_argument("--math-only", action="store_true")
    parser.add_argument("--terminal-prefix-only", action="store_true")
    args = parser.parse_args()
    OUT = OUT/f"validation-{time.time_ns()}"
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        if args.terminal_prefix_only:
            terminal_recurrent_prefix(args.compiler)
        else:
            for compiler in args.compiler:
                fixtures(compiler)
                attention_boundaries(compiler)
                if args.native_evidence:
                    saved_evidence(compiler,args.native_evidence)
            if not args.math_only:
                tiny_graph(args.compiler)
                terminal_recurrent_prefix(args.compiler)
    finally:
        (OUT/"cpu-tests.json").write_text(json.dumps(REPORT,indent=2)+"\n")
    print(f"PASS: {len(REPORT)} bounded CPU native-semantic checks; exact/tolerance: {OUT/'cpu-tests.json'}")


if __name__ == "__main__":
    main()
