"""Admission unit tests only: no model, CPU compiler, GPU or oracle launch."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import run_native_model as runner


class Admission(unittest.TestCase):
    def lease(self):
        return dict(authorized=True,worktree=str(runner.ROOT),compiler="flang",P=18,G=17,
                    full_reset=True,RSS_cap_bytes=runner.RSS_CAP,timeout_seconds=1800,GPU=False,
                    unit_test_only=True)

    def test_valid_and_fail_closed_lease(self):
        valid = self.lease()
        runner.validate_lease(valid,"flang")
        changes = dict(authorized=1,worktree="wrong-tree",compiler="gfortran",P=18.0,G=True,
                       full_reset=1,RSS_cap_bytes=True,timeout_seconds=float("nan"),GPU=0)
        for field, bad in changes.items():
            lease = {**valid,field:bad}
            with self.subTest(field=field), self.assertRaises(AssertionError):
                runner.validate_lease(lease,"flang")
        for field in valid.keys()-{"unit_test_only"}:
            lease = copy.copy(valid)
            del lease[field]
            with self.subTest(missing=field), self.assertRaises(KeyError):
                runner.validate_lease(lease,"flang")

    def test_lfortran_window_does_not_extend_old_compilers(self):
        lease = {**self.lease(),"compiler":"lfortran","timeout_seconds":3600}
        runner.validate_lease(lease,"lfortran")
        for compiler in ("flang","gfortran"):
            with self.subTest(compiler=compiler), self.assertRaises(AssertionError):
                runner.validate_lease({**lease,"compiler":compiler},compiler)
        with self.assertRaises(AssertionError):
            runner.validate_lease({**lease,"timeout_seconds":3601},"lfortran")

    def lfortran_fixture(self):
        base = (Path("artifacts/fortran-native")/f"lfortran-admission-unit-{time.time_ns()}").resolve()
        app, g, archive, binary = [base/name for name in ("app","g","g/source","g/binary")]
        contents = {"src-fortran/main.f90":b"committed main image\n",
                    "src-fortran/dense_kernels.f90":b"committed math image\n"}
        for root in (app,archive):
            for name, raw in contents.items():
                path = root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(raw)
        compiler = g/"build/llvm11/src/bin/lfortran"
        runtime = g/"build/llvm11/src/runtime/liblfortran_runtime.0.dylib"
        for path in (compiler,runtime,binary/"qwen-raw"):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b"identity fixture: "+path.name.encode())
        flags = ["--fast","--std=f23","--no-fast-math"]
        dep = dict(install_name="@rpath/liblfortran_runtime.0.dylib",path=str(runtime),
                   resolved_path=str(runtime.resolve()),sha256=runner.digest(runtime))
        runtime_receipt = g/"runtime-identity.json"
        runtime_receipt.write_text(json.dumps({"qwen-raw":dict(actual_lfortran_runtime=dep,
            actual_loaded_libraries=[dict(resolved_path=str(binary/"qwen-raw"),
                                          sha256=runner.digest(binary/"qwen-raw"))])})+"\n")
        build = dict(compiler="lfortran",compiler_worktree=str(g),pixi_environment="llvm11",
                     compiler_commit="d"*40,compiler_path=str(compiler),compiler_sha256=runner.digest(compiler),
                     source_commit=runner.LF_SOURCE_COMMIT,source_archive_root=str(archive),
                     source_sha256={name:runner.digest(archive/name) for name in contents},flags=flags,
                     binary_sha256={"qwen-raw":runner.digest(binary/"qwen-raw")},runtime_dependency=dep,
                     commands=[dict(command=["lfortran",*flags,"-c",str(archive/name),"-o",str(binary/(Path(name).stem+".o"))],
                                    returncode=0) for name in contents] +
                              [dict(command=["lfortran",*flags,str(binary/"main.o"),"-o",str(binary/"qwen-raw")],returncode=0)])
        build_path = g/"build-receipt.json"
        build_path.write_text(json.dumps(build)+"\n")
        h = {k:build[k] for k in ("compiler","compiler_worktree","pixi_environment","compiler_commit",
            "compiler_path","compiler_sha256","source_commit","source_archive_root","flags","binary_sha256","runtime_dependency")}
        h.update(build_receipt=str(build_path),build_receipt_sha256=runner.digest(build_path),
                 binary_root=str(binary),runtime_identity_receipt=str(runtime_receipt),
                 runtime_identity_receipt_sha256=runner.digest(runtime_receipt),C_computational_linkage=False)
        path = g/"handoff.json"
        path.write_text(json.dumps(h)+"\n")
        return dict(base=base,app=app,g=g,handoff=path,contents=contents,header=h,build=build,
                    build_path=build_path,archive=archive,binary=binary,compiler=compiler,runtime=runtime)

    def lfortran_context(self, f):
        def git(*command):
            if command[-1] == "HEAD":
                return f["header"]["compiler_commit"]+"\n"
            if "ls-tree" in command:
                return "\n".join(f["contents"])+"\n"
            return runner.LF_SOURCE_COMMIT+"\n"

        context = ExitStack()
        context.enter_context(patch.object(runner,"ROOT",f["app"]))
        context.enter_context(patch.object(runner,"LF_WORKTREE",f["g"]))
        context.enter_context(patch.object(runner.Path,"cwd",return_value=f["g"]))
        context.enter_context(patch.object(runner,"command_output",side_effect=git))
        context.enter_context(patch.object(runner.subprocess,"check_output",
                                          side_effect=lambda command:f["contents"][command[-1].split(":",1)[1]]))
        context.enter_context(patch.dict(runner.os.environ,CONDA_PREFIX=str(f["g"]/".pixi/envs/llvm11"),
            PIXI_PROJECT_ROOT=str(f["g"]),PIXI_ENVIRONMENT_NAME="llvm11",DYLD_LIBRARY_PATH="",
            DYLD_FALLBACK_LIBRARY_PATH="",DYLD_INSERT_LIBRARIES=""))
        return context

    def test_explicit_lfortran_binding_and_shared_supervisor(self):
        f = self.lfortran_fixture()
        artifacts = f["base"]/"receipts"
        artifacts.mkdir()
        lease = artifacts/"lease.json"
        lease.write_text(json.dumps({**self.lease(),"worktree":str(f["app"]),"compiler":"lfortran",
                                    "timeout_seconds":3600})+"\n")

        def fake_process(command, **unused):
            Path(command[command.index("--result")+1]).write_text(json.dumps(self.fake_result())+"\n")
            return SimpleNamespace(returncode=0,pid=0,poll=lambda:0)

        with self.lfortran_context(f), patch.object(runner,"ARTIFACTS",artifacts), \
                patch.object(runner,"preflight",side_effect=lambda compiler,handoff:
                    {**self.fake_receipt(),**runner.lfortran_build(handoff)}) as preflight, \
                patch.object(runner.subprocess,"Popen",side_effect=fake_process) as process, \
                patch.object(runner.resource,"getrusage",return_value=SimpleNamespace(ru_maxrss=1024)), \
                patch("sys.argv",["run_native_model.py","--compiler","lfortran","--lfortran-handoff",
                                  str(f["handoff"]),"--lease",str(lease)]):
            runner.main()
            preflight.assert_called_once_with("lfortran",f["handoff"])
            self.assertEqual(process.call_args.args[0][0],str(f["binary"]/"qwen-raw"))
        receipt = json.loads(next(artifacts.glob("model-*/completed.json")).read_text())
        self.assertEqual(receipt["compiler"],"lfortran")
        self.assertEqual(receipt["source_commit"],runner.LF_SOURCE_COMMIT)
        self.assertTrue(receipt["unit_test_only"] and receipt["resources_released"])

    def test_lfortran_stale_identity_and_missing_mapping_rejections(self):
        f = self.lfortran_fixture()
        with self.lfortran_context(f):
            runner.lfortran_build(f["handoff"])
            for path in (f["archive"]/"src-fortran/main.f90",f["app"]/"src-fortran/dense_kernels.f90",
                         f["binary"]/"qwen-raw",f["compiler"],f["runtime"]):
                original = path.read_bytes()
                path.write_bytes(original+b"stale")
                with self.subTest(stale=str(path)), self.assertRaises(AssertionError):
                    runner.lfortran_build(f["handoff"])
                path.write_bytes(original)
            for env in [dict(CONDA_PREFIX="wrong"),dict(PIXI_ENVIRONMENT_NAME="default"),
                        dict(PIXI_PROJECT_ROOT="wrong")]:
                with self.subTest(environment=env), patch.dict(runner.os.environ,env), self.assertRaises(AssertionError):
                    runner.lfortran_build(f["handoff"])
            for field, stale in [("source_commit","0"*40),("compiler_commit","e"*40)]:
                f["handoff"].write_text(json.dumps({**f["header"],field:stale})+"\n")
                with self.subTest(stale=field), self.assertRaises(AssertionError):
                    runner.lfortran_build(f["handoff"])
            for field in ("compiler_worktree","pixi_environment","binary_sha256","source_commit"):
                h = {k:v for k,v in f["header"].items() if k != field}
                f["handoff"].write_text(json.dumps(h)+"\n")
                with self.subTest(missing=field), self.assertRaises(KeyError):
                    runner.lfortran_build(f["handoff"])
            f["handoff"].write_text(json.dumps(f["header"])+"\n")
            build = copy.deepcopy(f["build"])
            del build["source_sha256"]["src-fortran/main.f90"]
            f["build_path"].write_text(json.dumps(build)+"\n")
            h = {**f["header"],"build_receipt_sha256":runner.digest(f["build_path"])}
            f["handoff"].write_text(json.dumps(h)+"\n")
            with self.assertRaisesRegex(AssertionError,"missing Fortran source mapping"):
                runner.lfortran_build(f["handoff"])
            build = copy.deepcopy(f["build"])
            build["binary_sha256"] = {}
            f["build_path"].write_text(json.dumps(build)+"\n")
            h = {**f["header"],"binary_sha256":{},"build_receipt_sha256":runner.digest(f["build_path"])}
            f["handoff"].write_text(json.dumps(h)+"\n")
            with self.assertRaisesRegex(AssertionError,"missing external binary mapping"):
                runner.lfortran_build(f["handoff"])
            build = copy.deepcopy(f["build"])
            build["commands"][0]["command"][1] = "--different-flags"
            f["build_path"].write_text(json.dumps(build)+"\n")
            f["handoff"].write_text(json.dumps({**f["header"],"build_receipt_sha256":runner.digest(f["build_path"])})+"\n")
            with self.assertRaisesRegex(AssertionError,"actual build flags changed"):
                runner.lfortran_build(f["handoff"])

    def fixture(self):
        artifacts = Path("artifacts/fortran-native")/f"admission-unit-{time.time_ns()}"
        artifacts.mkdir(parents=True)
        lease = artifacts/"lease.json"
        content = self.lease()
        content["worktree"] = str(artifacts.resolve()/"mock-worktree")
        lease.write_text(json.dumps(content)+"\n")
        return artifacts, lease

    def fake_receipt(self):
        return dict(binary="never-launched",input="never-read",expected_ids=list(range(17)),
                    expected_lookahead=17,unit_test_only=True,real_model_acceptance=False)

    def fake_result(self):
        return dict(schema=3,arithmetic_profile="native-semantic-fortran-cpu",
                    generated_ids=list(range(17)),consumed_decode_ids=list(range(17)),lookahead_id=17,
                    actual_generated=17,inference_cache_tokens_processed=35,inference_forward_calls=19,
                    cache_only_forward_calls=1,head_calls=18,sampler_calls=18,
                    extra_unemitted_decode_forwards=1,reset_cache_and_logits_exact=True,reset_forward_calls=19)

    def test_authorized_request_and_one_use_receipt(self):
        artifacts, lease = self.fixture()

        def fake_process(command, **unused):
            result = Path(command[command.index("--result")+1])
            result.write_text(json.dumps(self.fake_result())+"\n")
            return SimpleNamespace(returncode=0,pid=0,poll=lambda:0)

        with patch.object(runner,"ROOT",Path(json.loads(lease.read_text())["worktree"])), \
                patch.object(runner,"ARTIFACTS",artifacts), \
                patch.object(runner,"preflight",return_value=self.fake_receipt()), \
                patch.object(runner.subprocess,"Popen",side_effect=fake_process) as process, \
                patch.object(runner.resource,"getrusage",return_value=SimpleNamespace(ru_maxrss=1024)), \
                patch("sys.argv",["run_native_model.py","--lease",str(lease)]):
            runner.main()
            self.assertEqual(process.call_count,1)
            comparison = next(artifacts.glob("model-*/comparison.json"))
            self.assertTrue(all(json.loads(comparison.read_text())["checks"].values()))
            with self.assertRaises(FileExistsError):
                runner.main()
            self.assertEqual(process.call_count,1)
        receipts = [json.loads(p.read_text()) for p in artifacts.glob("model-*/completed.json")]
        self.assertEqual(sorted(r["status"] for r in receipts),["FAIL","PASS"])
        self.assertTrue(all(r["unit_test_only"] and r["resources_released"] for r in receipts))

    def test_missing_authorization_never_launches(self):
        with patch.object(runner.subprocess,"Popen") as process, \
                patch.object(runner,"preflight") as preflight, \
                patch("sys.argv",["run_native_model.py"]):
            with self.assertRaises(AssertionError):
                runner.main()
            process.assert_not_called()
            preflight.assert_not_called()

    def test_rss_live_values_and_child_exit_race(self):
        live = SimpleNamespace(pid=123,poll=lambda:None)
        exited = SimpleNamespace(pid=123,poll=lambda:0)
        for raw, expected in [("12345\n",12345*1024), (" 0\n",0)]:
            with self.subTest(valid=raw), patch.object(runner.subprocess,"run",
                    return_value=SimpleNamespace(returncode=0,stdout=raw)):
                self.assertEqual(runner.live_rss(live),expected)
        for raw in ["", "  \n", "invalid", "-1", "123\n456", "1.5"]:
            with self.subTest(invalid=raw), patch.object(runner.subprocess,"run",
                    return_value=SimpleNamespace(returncode=0,stdout=raw)):
                with self.assertRaisesRegex(RuntimeError,"missing/invalid RSS"):
                    runner.live_rss(live)
                self.assertIsNone(runner.live_rss(exited))
        with patch.object(runner.subprocess,"run",return_value=SimpleNamespace(returncode=1,stdout="")):
            with self.assertRaisesRegex(RuntimeError,"live child"):
                runner.live_rss(live)
            self.assertIsNone(runner.live_rss(exited))

    def test_wrong_ids_preserve_failed_receipt(self):
        artifacts, lease = self.fixture()

        def fake_process(command, **unused):
            wrong = self.fake_result()
            wrong["generated_ids"][3] = 999
            result = Path(command[command.index("--result")+1])
            result.write_text(json.dumps(wrong)+"\n")
            return SimpleNamespace(returncode=0,pid=0,poll=lambda:0)

        with patch.object(runner,"ROOT",Path(json.loads(lease.read_text())["worktree"])), \
                patch.object(runner,"ARTIFACTS",artifacts), \
                patch.object(runner,"preflight",return_value=self.fake_receipt()), \
                patch.object(runner.subprocess,"Popen",side_effect=fake_process), \
                patch.object(runner.resource,"getrusage",return_value=SimpleNamespace(ru_maxrss=1024)), \
                patch("sys.argv",["run_native_model.py","--lease",str(lease)]):
            with self.assertRaises(AssertionError):
                runner.main()
        comparison = json.loads(next(artifacts.glob("model-*/comparison.json")).read_text())
        self.assertEqual(comparison["first_ID_divergence"],3)
        self.assertFalse(comparison["checks"]["ids"])
        receipt = json.loads(next(artifacts.glob("model-*/completed.json")).read_text())
        self.assertEqual(receipt["status"],"FAIL")
        self.assertTrue(receipt["unit_test_only"] and receipt["resources_released"])


if __name__ == "__main__":
    unittest.main()
