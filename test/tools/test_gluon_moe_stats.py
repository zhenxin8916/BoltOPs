# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Reject performance comparisons without a single-TG model interval."""

import pytest
import json
from argparse import Namespace

from boltops.tools.gluon_moe_gemm1 import baseline_options, read_single_tg_stats, select_triton_config


@pytest.mark.parametrize("tokens,selected", [(64, 64), (2560, 2048)])
def test_json_baseline_preserves_all_settings_and_selects_nearest_m(tmp_path, tokens, selected):
    configs = {
        str(m):
        dict(BLOCK_SIZE_M=16 if m == 64 else 128, BLOCK_SIZE_N=32 if m == 64 else 256,
             BLOCK_SIZE_K=256 if m == 64 else 64, num_warps=2 if m == 64 else 8, num_stages=1, USE_MLS_LOAD=m == 64,
             use_block_pingpong=m != 64, N_TILES_PER_CTA=1, GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False, kpack=1,
             schedule_hint="none", sched_latency="none", buffer_cache_swizzle=False, c_sorted=True)
        for m in (64, 2048, 4096)
    }
    path = tmp_path / "mr149.json"
    path.write_text(json.dumps(configs))
    selection = select_triton_config(path, tokens)
    args = Namespace(bm=None, bn=None, triton_bk=None, triton_waves=None, triton_stages=None, triton_async=None,
                     triton_mls=None, triton_pingpong=None)
    options = baseline_options(args, selection)
    assert selection["selected_M"] == selected
    assert options == configs[str(selected)]
    assert (args.bm, args.bn) == (options["BLOCK_SIZE_M"], options["BLOCK_SIZE_N"])


@pytest.mark.parametrize(
    "override", ["triton_bk", "triton_waves", "triton_stages", "triton_async", "triton_mls", "triton_pingpong", "bn"])
def test_json_baseline_rejects_silent_overrides_and_different_output_tiles(override):
    args = Namespace(bm=None, bn=None, triton_bk=None, triton_waves=None, triton_stages=None, triton_async=None,
                     triton_mls=None, triton_pingpong=None)
    setattr(args, override, 64 if override == "bn" else 1)
    with pytest.raises(ValueError, match="manual|match JSON"):
        baseline_options(args, {"options": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32}})


@pytest.mark.parametrize("setting", [{"a_sorted": True}, {"c_sorted": False}, {"ck_sorting": False}, {"N_TILES_PER_CTA": 2}])
def test_json_baseline_rejects_unsupported_route_contract(tmp_path, setting):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"64": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=256, num_warps=2, num_stages=1, **setting)}))
    with pytest.raises(ValueError, match="c_sorted=True"):
        select_triton_config(path, 64)


@pytest.mark.parametrize("recipe", ["mxfp8_scaled", "fp16"])
def test_channel_json_cannot_be_used_for_another_recipe(tmp_path, recipe):
    with pytest.raises(ValueError, match="FP8 channel only"):
        select_triton_config(tmp_path / "channel.json", 64, recipe=recipe)


def write_stats(root, *, groups=1, start=1000, end=3500):
    target = root / "m5out/0/0/stats.txt"
    target.parent.mkdir(parents=True)
    target.write_text("---------- Begin Simulation Statistics ----------\n"
                      "simFreq 1000000000000\n"
                      "system.simTicks 999999999\n"
                      f"system.shaders.firstWaveStartTick {start}\n"
                      f"system.shaders.lastWaveEndTick {end}\n"
                      f"system.shaders.CUs00.completedWGs {groups}\n"
                      "system.shaders.CUs01.completedWGs 0\n")
    return target


def test_uses_tg_interval_not_simulation_wall_time(tmp_path):
    write_stats(tmp_path)
    result = read_single_tg_stats(tmp_path)
    assert result["elapsed_ticks"] == 2500
    assert result["time_ns"] == 2.5
    assert result["completed_tgs"] == 1


@pytest.mark.parametrize("groups", [0, 2, 0.5])
def test_rejects_non_single_tg_results(tmp_path, groups):
    write_stats(tmp_path, groups=groups)
    with pytest.raises(ValueError, match="one completed TG"):
        read_single_tg_stats(tmp_path)


def test_rejects_ambiguous_dispatch(tmp_path):
    first = write_stats(tmp_path)
    second = tmp_path / "m5out/0/1/stats.txt"
    second.parent.mkdir()
    second.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="one dispatch"):
        read_single_tg_stats(tmp_path)


def test_process_aggregate_does_not_count_as_another_dispatch(tmp_path):
    write_stats(tmp_path)
    (tmp_path / "m5out/stats.txt").write_text("totalSimTicks 999999999\n")
    assert read_single_tg_stats(tmp_path)["time_ns"] == 2.5


def test_rejects_invalid_interval(tmp_path):
    write_stats(tmp_path, start=3500, end=1000)
    with pytest.raises(ValueError, match="invalid TG timing"):
        read_single_tg_stats(tmp_path)


def test_rejects_nonfinite_timing(tmp_path):
    path = write_stats(tmp_path)
    path.write_text(path.read_text().replace("lastWaveEndTick 3500", "lastWaveEndTick 1e999"))
    with pytest.raises(ValueError, match="invalid TG timing"):
        read_single_tg_stats(tmp_path)


@pytest.mark.parametrize("corruption", [None, "input", "stats", "missing_validation", "empty_validation"])
def test_report_checks_pair_inputs_and_raw_stats(tmp_path, corruption):
    from boltops.tools.gluon_moe_report import load_experiment

    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "gemm": "gemm1",
            "case": {"id": "single-tg-fixture"},
            "recipe": "fp16",
            "config": {},
        }))
    results = {}
    for mode in ("non_wasp", "wasp"):
        directory = tmp_path / mode
        write_stats(directory)
        (directory / "validation.json").write_text(json.dumps({
            "checked_elements": 16, "mismatched_elements": 0, "nonfinite_elements": 0,
            "max_absolute_error": 0,
        }))
        results[mode] = {
            "status": "passed", "route": {"input_sha256": {"A": "same-input"}}, "timing":
            read_single_tg_stats(directory)
        }
    if corruption == "missing_validation":
        (tmp_path / "wasp/validation.json").unlink()
    if corruption == "empty_validation":
        path = tmp_path / "wasp/validation.json"
        validation = json.loads(path.read_text())
        validation["checked_elements"] = 0
        path.write_text(json.dumps(validation))
    if corruption == "input":
        results["wasp"]["route"]["input_sha256"]["A"] = "different-input"
    if corruption == "stats":
        path = tmp_path / "wasp/m5out/0/0/stats.txt"
        path.write_text(path.read_text().replace("lastWaveEndTick 3500", "lastWaveEndTick 4500"))
    (tmp_path / "summary.json").write_text(json.dumps(results))
    record = load_experiment(tmp_path)
    assert record["status"] == ("passed" if corruption is None else "invalid")
    assert bool(record["speedups"]) == (corruption is None)


def test_report_distinguishes_ordinary_triton_from_gluon_non_wasp(tmp_path):
    from boltops.tools.gluon_moe_report import load_experiment

    modes = ["triton", "non_wasp", "wasp"]
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "gemm": "gemm1",
            "case": {"id": "baseline-fixture"},
            "recipe": "fp8_channel",
            "config": {},
            "modes": modes,
        }))
    results = {}
    for mode, end in zip(modes, [5000, 6000, 3000]):
        directory = tmp_path / mode
        write_stats(directory, end=end)
        (directory / "validation.json").write_text(json.dumps({
            "checked_elements": 16, "mismatched_elements": 0, "nonfinite_elements": 0,
            "max_absolute_error": 0,
        }))
        results[mode] = {
            "status": "passed", "route": {"input_sha256": {"A": "same-input"}}, "timing":
            read_single_tg_stats(directory)
        }
    (tmp_path / "summary.json").write_text(json.dumps(results))
    record = load_experiment(tmp_path)
    assert record["status"] == "passed"
    assert record["speedups"]["wasp"] == 2.5
    assert record["speedups_vs_triton"]["wasp"] == 2
    assert record["speedups_vs_triton"]["non_wasp"] == 0.8
    # A baseline-only search has no Gluon result and must not invent one.
    (tmp_path / "summary.json").write_text(json.dumps({"triton": results["triton"]}))
    assert load_experiment(tmp_path)["status"] == "incomplete"


@pytest.mark.parametrize("corruption", [None, "snapshot", "launch", "metadata", "tile", "recipe"])
def test_report_verifies_json_baseline_provenance(tmp_path, corruption):
    from boltops.tools.gluon_moe_gemm1 import digest_file
    from boltops.tools.gluon_moe_report import load_experiment

    config = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=256, num_warps=2, num_stages=1, USE_MLS_LOAD=True)
    snapshot = tmp_path / "triton-baseline.json"
    snapshot.write_text(json.dumps({"64": config}))
    manifest = dict(
        gemm="gemm1", case={"id": "mr149-fixture"}, recipe="fp8_channel", modes=["triton", "wasp"],
        config={"bm": 16, "bn": 32}, triton_baseline={
            "options": config, "json_config": {"selected_M": 64, "sha256": digest_file(snapshot), "options": config}
        })
    results = {}
    for mode in manifest["modes"]:
        directory = tmp_path / mode
        write_stats(directory)
        (directory / "validation.json").write_text(json.dumps({
            "checked_elements": 16, "mismatched_elements": 0, "nonfinite_elements": 0,
            "max_absolute_error": 0,
        }))
        results[mode] = {
            "status": "passed", "route": {"input_sha256": {"A": "same"}}, "timing": read_single_tg_stats(directory)
        }
    launch, metadata = config.copy(), {"num_warps": 2, "num_stages": 1}
    if corruption == "snapshot":
        snapshot.write_text("{}")
    if corruption == "launch":
        launch["USE_MLS_LOAD"] = False
    if corruption == "metadata":
        metadata["num_stages"] = 2
    if corruption == "tile":
        manifest["config"]["bn"] = 64
    if corruption == "recipe":
        manifest["recipe"] = "mxfp8_scaled"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps(results))
    (tmp_path / "triton/launch.json").write_text(json.dumps({"options": launch}))
    (tmp_path / "triton/compiled.json").write_text(json.dumps({"kernel_metadata": metadata}))
    record = load_experiment(tmp_path)
    assert record["status"] == ("passed" if corruption is None else "invalid")
    assert bool(record["speedups_vs_triton"]) == (corruption is None)


def test_report_indexes_experiments_without_name_prefix(tmp_path, monkeypatch):
    from boltops.tools.gluon_moe_report import main

    for name in ("final-qwen-channel", "confirmed-qwen-mx"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({
            "gemm": "gemm1", "case": {"id": name}, "recipe": "fp8_channel",
            "config": {}, "modes": ["wasp"],
        }))
    (tmp_path / "unrelated").mkdir()
    out = tmp_path / "report"
    monkeypatch.setattr("sys.argv", ["gluon_moe_report", "--root", str(tmp_path), "--out", str(out)])
    main()
    records = json.loads((out / "index.json").read_text())["experiments"]
    assert {record["id"] for record in records} == {"final-qwen-channel", "confirmed-qwen-mx"}
    assert all(record["status"] == "incomplete" for record in records)


def test_snapshot_excludes_model_outputs(tmp_path):
    import subprocess
    from boltops.tools.gluon_moe_gemm1 import snapshot_repo

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "--allow-empty", "-m", "base"], check=True, capture_output=True)
    (repo / "kernel.py").write_text("# untracked kernel\n")
    (repo / "config.json").write_text("{}\n")
    (repo / "m5out").mkdir()
    (repo / "m5out/stats.txt").write_text("model output\n")
    snapshot = tmp_path / "snapshot"
    version = snapshot_repo(repo, snapshot)
    assert set(version["changed_source_sha256"]) == {"kernel.py", "config.json"}
    assert not (snapshot / "changed_sources/m5out").exists()


@pytest.mark.parametrize("mode", ["triton", "wasp"])
@pytest.mark.parametrize("bk", [64, 128])
def test_channel_cli_keeps_triton_bk_tunable_and_limits_gluon(tmp_path, monkeypatch, mode, bk):
    from boltops.tools import gluon_moe_gemm1 as driver

    launches = []
    monkeypatch.setattr(driver, "worker", lambda args, case: launches.append(args.bk))
    monkeypatch.setattr("sys.argv", [
        "gluon_moe_gemm1", "--worker", mode, "--modes", mode,
        "--recipe", "fp8_channel", "--bk", str(bk), "--out", str(tmp_path),
    ])
    if mode != "triton" and bk != 128:
        with pytest.raises(SystemExit):
            driver.main()
        assert launches == []
    else:
        driver.main()
        assert launches == [bk]


@pytest.mark.parametrize("bn,regs", [(128, [32, 48, 216, 216]), (256, [32, 48, 216, 216])])
def test_default_gluon_probe_uses_best_checkpoint_config(tmp_path, monkeypatch, bn, regs):
    from boltops.tools import gluon_moe_gemm1 as driver
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import DEFAULT_LAUNCH_OPTIONS

    launches = []
    monkeypatch.setattr(driver, "worker", lambda args, case: launches.append(args))
    monkeypatch.setattr("sys.argv", ["gluon_moe_gemm1", "--worker", "wdra", "--recipe", "fp8_channel", "--bn", str(bn), "--out", str(tmp_path)])
    driver.main()
    args, = launches
    assert (args.bm, args.bn, args.bk) == (128, bn, 128)
    assert args.regs == regs
    assert args.mls_tile == (64, 32)
    assert DEFAULT_LAUNCH_OPTIONS["wasp_wdra"]
    assert "amdgpu_enable_max_ilp_scheduling_strategy" not in DEFAULT_LAUNCH_OPTIONS


def test_cpu_inputs_use_ck_packed_routes():
    import torch
    from boltops.tools.gluon_moe_gemm1 import prepare_cpu

    tensors, _, _, _, metadata = prepare_cpu(dict(M=17, E=2, N=32, K=512, topk=2),
                                             "fp8_channel", 128, 128, 0, 3101)
    packed = tensors["Sorted"].to(torch.int64)
    tokens, choices = packed & 0x00FFFFFF, (packed >> 24) & 0xFF
    valid = tokens < 17
    assert (tokens[~valid] == 17).all()
    assert choices[valid].max() == 1
    flattened = tokens[valid] * 2 + choices[valid]
    torch.testing.assert_close(tensors["SortedWeights"][valid], tensors["RouteWeights"][flattened])
    assert "CK" in metadata["sorting_format"]


def test_probe_defaults_gc1800_and_preserves_other_model_args(tmp_path, monkeypatch):
    import ast
    import os
    from boltops.tools import gluon_moe_gemm1 as driver
    launches=[]
    monkeypatch.setenv("GPU_ARGS", "['--gpu-clock=2.2GHz', '--umc-bw=1.8TB/s']")
    monkeypatch.setattr(driver,"worker",lambda args,case: launches.append((args.gc_ghz,ast.literal_eval(os.environ["GPU_ARGS"]))))
    monkeypatch.setattr("sys.argv",["gluon_moe_gemm1","--worker","wdra","--recipe","fp8_channel","--out",str(tmp_path)])
    driver.main()
    assert launches==[(1.8,['--umc-bw=1.8TB/s','--gpu-clock=1.8GHz'])]
