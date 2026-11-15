"""Lease-gated, bounded CPU acceptance against immutable native short IDs."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT/"artifacts/fortran-native"
MODEL = Path("/Users/ondrej/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit")
RSS_CAP = 38*1024**3
TIMEOUT = 1800
LF_TIMEOUT = 3600
LF_SOURCE_COMMIT = "c077fe0263e054dcbdde70808f938c8b71b9f3c5"
LF_WORKTREE = Path("/Users/ondrej/repos/lfortran-7/.fix-issue/qwen38-compiler/worktrees/metal-precision")
if not __debug__:
    raise RuntimeError("CPU model safety checks require non-optimized Python through the declared Pixi task")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_output(*command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def validate_lease(lease, compiler):
    assert compiler in ("flang","gfortran","lfortran")
    assert lease["authorized"] is True and lease["worktree"] == str(ROOT)
    assert lease["compiler"] == compiler
    assert type(lease["P"]) is int and lease["P"] == 18
    assert type(lease["G"]) is int and lease["G"] == 17
    assert lease["full_reset"] is True and lease["GPU"] is False
    assert type(lease["RSS_cap_bytes"]) is int and 0 < lease["RSS_cap_bytes"] <= RSS_CAP
    timeout = lease["timeout_seconds"]
    limit = LF_TIMEOUT if compiler == "lfortran" else TIMEOUT
    assert type(timeout) in (int,float) and math.isfinite(timeout) and 0 < timeout <= limit


def lfortran_build(handoff_path):
    assert handoff_path is not None and handoff_path.is_absolute(), "absolute LFortran handoff required"
    h = json.loads(handoff_path.read_text())
    assert h["compiler"] == "lfortran" and h["pixi_environment"] == "llvm11"
    g = Path(h["compiler_worktree"])
    assert g.is_absolute() and g.resolve() == LF_WORKTREE.resolve(), "unexpected LFortran worktree"
    assert Path.cwd().resolve() == g.resolve(), "run from the bound compiler worktree through Pixi"
    assert os.environ.get("CONDA_PREFIX") == str(g/".pixi/envs/llvm11"), "active G LLVM11 environment required"
    assert os.environ.get("PIXI_PROJECT_ROOT") == str(g) and os.environ.get("PIXI_ENVIRONMENT_NAME") == "llvm11"
    paths = {k:Path(h[k]) for k in ("compiler_path","source_archive_root","build_receipt","binary_root")}
    for path in paths.values():
        assert path.is_absolute() and path.resolve().is_relative_to(g.resolve()), "external path outside G"
    compiler = paths["compiler_path"]
    assert compiler.resolve() == (g/"build/llvm11/src/bin/lfortran").resolve()
    assert digest(compiler) == h["compiler_sha256"], "stale external compiler"
    assert command_output("git","-C",str(g),"rev-parse","HEAD").strip() == h["compiler_commit"], "stale compiler commit"
    assert h["source_commit"] == LF_SOURCE_COMMIT, "unexpected archived application commit"
    assert command_output("git","-C",str(ROOT),"rev-parse",LF_SOURCE_COMMIT+"^{commit}").strip() == LF_SOURCE_COMMIT
    assert digest(paths["build_receipt"]) == h["build_receipt_sha256"], "stale external build receipt"
    build = json.loads(paths["build_receipt"].read_text())
    for key in ("compiler","compiler_worktree","pixi_environment","compiler_path","compiler_commit",
                "compiler_sha256","source_commit","flags","binary_sha256","source_archive_root"):
        assert build[key] == h[key], "handoff/build receipt disagreement: "+key
    mapping = build["source_sha256"]
    assert isinstance(mapping,dict) and mapping, "missing archived source mapping"
    committed = command_output("git","-C",str(ROOT),"ls-tree","-r","--name-only",LF_SOURCE_COMMIT,
                               "--","src-fortran").splitlines()
    required = {name for name in committed if name.endswith(".f90")}
    current = {str(p.relative_to(ROOT)) for p in (ROOT/"src-fortran").rglob("*.f90")}
    archived = {str(p.relative_to(paths["source_archive_root"]))
                for p in (paths["source_archive_root"]/"src-fortran").rglob("*.f90")}
    assert required and required == current == archived and required <= mapping.keys(), "missing Fortran source mapping"
    for name, sha in mapping.items():
        relative = Path(name)
        assert not relative.is_absolute() and relative.parts[0] == "src-fortran" and ".." not in relative.parts
        archived_file = paths["source_archive_root"]/relative
        assert archived_file.resolve().is_relative_to(paths["source_archive_root"].resolve())
        assert digest(archived_file) == sha, "stale archived source: "+name
        blob = subprocess.check_output(["git","-C",str(ROOT),"show",LF_SOURCE_COMMIT+":"+name])
        assert hashlib.sha256(blob).hexdigest() == sha, "archive differs from committed source: "+name
        if name in required:
            assert digest(ROOT/relative) == sha, "current FN Fortran source changed: "+name
    binaries = h["binary_sha256"]
    assert isinstance(binaries,dict) and "qwen-raw" in binaries, "missing external binary mapping"
    for name, sha in binaries.items():
        assert Path(name).name == name and digest(paths["binary_root"]/name) == sha, "stale external binary"
    runtime = h["runtime_dependency"]
    for key in ("install_name","path","resolved_path","sha256"):
        assert build["runtime_dependency"][key] == runtime[key], "runtime receipt disagreement"
    runtime_path = Path(runtime["path"])
    assert runtime_path.is_absolute() and runtime_path.resolve().is_relative_to(g.resolve())
    assert str(runtime_path.resolve()) == runtime["resolved_path"] and digest(runtime_path) == runtime["sha256"]
    runtime_receipt = Path(h["runtime_identity_receipt"])
    assert runtime_receipt.is_absolute() and runtime_receipt.resolve().is_relative_to(g.resolve())
    assert digest(runtime_receipt) == h["runtime_identity_receipt_sha256"], "stale loaded-runtime receipt"
    observed = json.loads(runtime_receipt.read_text())["qwen-raw"]
    assert observed["actual_lfortran_runtime"]["resolved_path"] == runtime["resolved_path"]
    assert observed["actual_lfortran_runtime"]["sha256"] == runtime["sha256"], "loaded runtime disagreement"
    assert any(v.get("resolved_path") == str(paths["binary_root"]/"qwen-raw") and
               v.get("sha256") == binaries["qwen-raw"] for v in observed["actual_loaded_libraries"])
    assert h["C_computational_linkage"] is False
    assert not any(os.environ.get(key) for key in
                   ("DYLD_LIBRARY_PATH","DYLD_FALLBACK_LIBRARY_PATH","DYLD_INSERT_LIBRARIES")), "DYLD override present"
    flags = h["flags"]
    assert isinstance(flags,list) and flags and all(isinstance(flag,str) for flag in flags)
    assert "--no-fast-math" in flags and "--fast-math" not in flags
    compiled = set()
    linked = False
    for record in build["commands"]:
        command = record["command"]
        assert type(record["returncode"]) is int and record["returncode"] == 0, "failed external build command"
        assert command[0] in ("lfortran",str(compiler)) and command[1:1+len(flags)] == flags, "actual build flags changed"
        if "-c" in command:
            inputs = [Path(v) for v in command if v.endswith(".f90")]
            assert len(inputs) == 1 and inputs[0].is_relative_to(paths["source_archive_root"])
            name = str(inputs[0].relative_to(paths["source_archive_root"]))
            assert name in required, "unmapped compiled Fortran source"
            compiled.add(name)
        linked |= command[-2:] == ["-o",str(paths["binary_root"]/"qwen-raw")]
    assert {name for name in required if Path(name).parent == Path("src-fortran")} <= compiled and linked
    return dict(compiler="lfortran",compiler_worktree=str(g),compiler_path=str(compiler),
                compiler_commit=h["compiler_commit"],compiler_sha256=h["compiler_sha256"],
                compiler_flags=flags,pixi_environment="llvm11",source_commit=LF_SOURCE_COMMIT,
                source_archive_root=str(paths["source_archive_root"]),archived_source_sha256=mapping,
                build_receipt=str(paths["build_receipt"]),build_receipt_sha256=h["build_receipt_sha256"],
                binary=str(paths["binary_root"]/"qwen-raw"),binary_sha256=binaries["qwen-raw"],
                binary_root=str(paths["binary_root"]),external_binary_sha256=binaries,
                runtime_dependency=runtime,runtime_identity_receipt=str(runtime_receipt),
                runtime_identity_receipt_sha256=h["runtime_identity_receipt_sha256"],
                lfortran_handoff=str(handoff_path),lfortran_handoff_sha256=digest(handoff_path))


def live_rss(process):
    ps = subprocess.run(["ps","-o","rss=","-p",str(process.pid)],text=True,capture_output=True)
    if process.poll() is not None:
        return None
    if ps.returncode:
        raise RuntimeError("CPU model RSS monitor failed for a live child")
    raw = ps.stdout.strip()
    if re.fullmatch(r"[0-9]+",raw) is None:
        raise RuntimeError("CPU model RSS monitor returned missing/invalid RSS for a live child")
    return int(raw)*1024


def preflight(compiler, lfortran_handoff=None):
    external = lfortran_build(lfortran_handoff) if compiler == "lfortran" else None
    prereg = ROOT.parent/"mlx-native/artifacts/native-benchmark-prep/preregistration.json"
    reference = json.loads(prereg.read_text())
    workload = reference["workloads"]["short_chat"]
    input_path, expected_path = Path(workload["input"]), Path(workload["expected"])
    assert digest(input_path) == workload["input_sha256"], "input identity changed"
    assert digest(expected_path) == workload["expected_sha256"], "native expected IDs changed"
    identity_path = Path(reference["model_identity"])
    assert digest(identity_path) == reference["model_identity_sha256"], "model identity receipt changed"
    identity = json.loads(identity_path.read_text())
    for name, pinned in identity["files_stat"].items():
        path = MODEL/name
        stat = path.stat()
        assert stat.st_size == pinned["size"] and stat.st_mtime_ns == pinned["mtime_ns"], name
        if name.endswith(".safetensors"):
            header = identity["shards"][name]
            with path.open("rb") as stream:
                prefix = stream.read(8)
                count = int.from_bytes(prefix,"little")
                assert count == header["header_bytes"], name
                assert hashlib.sha256(stream.read(count)).hexdigest() == header["header_sha256"], name
        else:
            assert digest(path) == identity["sha256"][name], name
    inputs = json.loads(input_path.read_text())
    expected = json.loads(expected_path.read_text())
    if isinstance(inputs, dict):
        inputs = inputs["input_ids"]
    if isinstance(expected, dict):
        expected = expected["generated_ids"]
    assert len(inputs) == 18 and len(expected) >= 18
    # Metadata-only: no weight read/hash/copy and no model initialization.
    config = MODEL/"config.json"
    cfg = json.loads(config.read_text())
    text = cfg.get("text_config",cfg)
    assert text["hidden_size"] == 5120 and text["intermediate_size"] == 17408
    assert text["num_hidden_layers"] == 64 and text["head_dim"] == 256
    assert cfg.get("model_type") in ("qwen3_5","qwen3_5_text")
    binary = Path(external["binary"]) if external else ROOT/f"build/fortran-driver/{compiler}/qwen-raw"
    assert binary.is_file(), "build the selected compiler through Pixi first"
    pressure = command_output("memory_pressure","-Q")
    free = int(re.search(r"System-wide memory free percentage:\s*(\d+)%",pressure)[1])
    assert free >= 55, f"insufficient reported memory headroom: {pressure}"
    power = command_output("pmset","-g","batt")
    assert "AC Power" in power, "AC power required"
    disk = shutil.disk_usage(ROOT).free
    assert disk >= 8*1024**3, "insufficient free disk"
    retained = sum(p.stat().st_size for p in ARTIFACTS.rglob("*") if p.is_file())
    assert retained < 4*1024**3, "artifact budget exceeded"
    environment = os.environ.get("CONDA_PREFIX","")
    if not external:
        assert str(ROOT/".pixi/envs/fortran-precision") == environment, "selected Pixi environment required"
    sources = {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/"src-fortran").glob("*.f90")}
    app_commit = command_output("git","-C",str(ROOT),"rev-parse","HEAD").strip()
    receipt = dict(worktree=str(ROOT),compiler=compiler,source_commit=app_commit,
                   sources=sources,binary=str(binary),binary_sha256=digest(binary),
                   application_worktree_commit=app_commit,execution_worktree=str(Path.cwd().resolve()),
                   runner_source=str(Path(__file__).resolve()),runner_sha256=digest(Path(__file__)),
                   input=str(input_path),input_sha256=digest(input_path),expected=str(expected_path),
                   expected_sha256=digest(expected_path),preregistration_sha256=digest(prereg),
                   model=str(MODEL),config_sha256=digest(config),model_identity=str(identity_path),
                   model_identity_sha256=digest(identity_path),P=18,G=17,full_reset=True,
                   peak_RSS_limit=RSS_CAP,timeout_seconds=LF_TIMEOUT if external else TIMEOUT,disk_free=disk,
                   memory_free_percent=free,minimum_reported_free_percent=55,
                   memory_percentage_scope="entry admission signal, not measured physical residency",
                   AC=power,performance_acceptance=False,
                   expected_ids=expected[:17],expected_lookahead=expected[17])
    if external:
        receipt.update(external)
    else:
        build = json.loads((binary.parent/"build-manifest.json").read_text())
        assert build["sources"] == sources and build["binary_sha256"] == digest(binary), "stale CPU model binary"
        receipt.update(build_manifest_sha256=digest(binary.parent/"build-manifest.json"),
                       compiler_identity=(binary.parent/"compiler.txt").read_text(),
                       compiler_flags=(binary.parent/"flags.txt").read_text())
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler",choices=["flang","gfortran","lfortran"],default="flang")
    parser.add_argument("--lfortran-handoff",type=Path)
    parser.add_argument("--lease",type=Path)
    parser.add_argument("--preflight-only",action="store_true")
    args = parser.parse_args()
    assert (args.compiler == "lfortran") == (args.lfortran_handoff is not None), "LFortran requires its explicit handoff"
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    if args.preflight_only:
        receipt = preflight(args.compiler,args.lfortran_handoff) if args.compiler == "lfortran" else preflight(args.compiler)
        print(json.dumps(receipt,indent=2))
        return
    assert args.lease is not None, "explicit parent CPU model lease required"
    lease = json.loads(args.lease.read_text())
    validate_lease(lease,args.compiler)
    receipt = preflight(args.compiler,args.lfortran_handoff) if args.compiler == "lfortran" else preflight(args.compiler)
    session = ARTIFACTS/f"model-{args.compiler}-{time.time_ns()}"
    session.mkdir()
    result = session/"result.json"
    command = [receipt["binary"],"--model",str(MODEL),"--input-ids",receipt["input"],
               "--max-tokens","17","--fixed-steps","--reset-check","--result",str(result)]
    receipt["command"] = command
    receipt["lease"] = str(args.lease.resolve())
    receipt["lease_sha256"] = digest(args.lease)
    (session/"handoff.json").write_text(json.dumps(receipt,indent=2)+"\n")
    started = time.monotonic()
    peak = 0
    process = None
    try:
        with (ARTIFACTS/"model-process.lock").open("a") as lock, (session/"run.log").open("wb") as log:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            used = ARTIFACTS/("model-lease-"+receipt["lease_sha256"]+".used")
            with used.open("x") as receipt_file:
                receipt_file.write(str(session)+"\n")
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            (session/"running.json").write_text(json.dumps(dict(pid=process.pid,session=str(session)))+"\n")
            with (session/"resources.jsonl").open("w") as samples:
                while process.poll() is None:
                    rss = live_rss(process)
                    if rss is None:
                        break
                    peak = max(peak,rss)
                    elapsed = time.monotonic()-started
                    samples.write(json.dumps(dict(seconds=elapsed,RSS=rss))+"\n")
                    samples.flush()
                    if rss > lease["RSS_cap_bytes"] or elapsed > lease["timeout_seconds"]:
                        raise RuntimeError("CPU model memory/time safety limit exceeded")
                    time.sleep(2)
            assert process.returncode == 0, f"model status {process.returncode}: {session/'run.log'}"
            peak = max(peak,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            assert peak <= lease["RSS_cap_bytes"], "CPU model actual high-water RSS exceeded cap"
            actual = json.loads(result.read_text())
            # Functional target is pinned native IDs, never scalar-C tokens.
            fields = ["schema","lookahead_id","actual_generated","inference_cache_tokens_processed",
                      "inference_forward_calls","cache_only_forward_calls","head_calls","sampler_calls",
                      "extra_unemitted_decode_forwards","reset_forward_calls"]
            checks = dict(integer_schema=all(type(actual[k]) is int for k in fields) and
                          all(type(v) is int for k in ("generated_ids","consumed_decode_ids") for v in actual[k]),
                          schema=actual["schema"] == 3 and actual["arithmetic_profile"] == "native-semantic-fortran-cpu",
                          ids=actual["generated_ids"] == receipt["expected_ids"],
                          lookahead=actual["lookahead_id"] == receipt["expected_lookahead"],
                          consumed=actual["consumed_decode_ids"] == receipt["expected_ids"],
                          emitted=actual["actual_generated"] == 17,
                          context=actual["inference_cache_tokens_processed"] == 35,
                          forwards=actual["inference_forward_calls"] == 19,
                          prefix=actual["cache_only_forward_calls"] == 1,
                          heads=actual["head_calls"] == 18,samplers=actual["sampler_calls"] == 18,
                          extra=actual["extra_unemitted_decode_forwards"] == 1,
                          reset=actual["reset_cache_and_logits_exact"] is True and actual["reset_forward_calls"] == 19)
            (session/"comparison.json").write_text(json.dumps(dict(checks=checks,
                first_ID_divergence=next((i for i,(a,b) in enumerate(zip(actual["generated_ids"],receipt["expected_ids"]))
                                          if a != b),None)),indent=2)+"\n")
            assert all(checks.values()), f"native short acceptance failed: {session/'comparison.json'}"
            receipt["status"] = "PASS"
    except BaseException as error:
        receipt["status"] = "FAIL"
        receipt["error"] = repr(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        receipt.update(peak_RSS=peak,elapsed_seconds=time.monotonic()-started,resources_released=True)
        (session/"completed.json").write_text(json.dumps(receipt,indent=2)+"\n")
        scope = "ADMISSION UNIT MOCK" if receipt.get("unit_test_only") else "CPU MODEL"
        print(f"{scope} {receipt.get('status','FAIL')}: {session}; owned resources released",flush=True)


if __name__ == "__main__":
    main()
