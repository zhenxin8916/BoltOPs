# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Single-TG WASP GEMM1 and ordinary Triton probes; execute in zx-perf-node60.

Example (after building the paired Triton worktree in the container):
  python -m boltops.tools.gluon_moe_gemm1 --case qwen35-small-m64 \
    --recipe fp8_channel --out /zhenxin/temp/gluon-qwen35-channel-001

Each mode is a fresh process and issues exactly one GPU kernel. CPU data,
quantization and reference are identical across modes. No timings are inferred
from host wall time. A result is ranked only after CPU validation and a stats
check proving exactly one completed workgroup in exactly one dispatch.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

CASES = Path(__file__).resolve().parents[1] / "fused_moe/gluon/qwen_cases.json"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def digest_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def select_triton_config(path, tokens, *, recipe="fp8_channel"):
    """Apply the ordinary MoE dispatcher's nearest-M rule without retuning."""
    if recipe != "fp8_channel":
        raise ValueError("this JSON baseline supports FP8 channel only; other recipes require their own configuration")
    configs = json.loads(path.read_text())
    if not configs or any(not key.isdecimal() or int(key) <= 0 for key in configs):
        raise ValueError("Triton config JSON must contain positive integer M keys")
    selected = min(configs, key=lambda key: abs(int(key) - tokens))
    config = configs[selected]
    for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"Triton config requires a positive integer {key}")
    # A is route-indexed; both comparison kernels write sorted C rows.
    # Reject incompatible configs instead of quietly dropping their settings.
    if (config.get("a_sorted") or not config.get("c_sorted", True)
            or not config.get("ck_sorting", True) or config.get("N_TILES_PER_CTA", 1) != 1):
        raise ValueError("single-TG comparison requires a_sorted=False, c_sorted=True, ck_sorting=True and N_TILES_PER_CTA=1")
    return {
        "path": str(path.resolve()), "sha256": digest_file(path), "selection_rule": "nearest_M", "requested_M": tokens,
        "selected_M": int(selected), "options": config
    }


def baseline_options(args, selection=None):
    if selection is not None:
        if any(
                getattr(args, name) is not None for name in ("triton_bk", "triton_waves", "triton_stages",
                                                             "triton_async", "triton_mls", "triton_pingpong")):
            raise ValueError("JSON baseline cannot be combined with manual --triton-* overrides")
        config = selection["options"].copy()
        for name, key in (("bm", "BLOCK_SIZE_M"), ("bn", "BLOCK_SIZE_N")):
            requested = getattr(args, name)
            if requested is not None and requested != config[key]:
                raise ValueError(f"paired Gluon {name.upper()} must match JSON {key}={config[key]}")
            setattr(args, name, config[key])
        return config
    return dict(BLOCK_SIZE_M=args.bm, BLOCK_SIZE_N=args.bn,
                BLOCK_SIZE_K=args.triton_bk if args.triton_bk is not None else args.bk, GROUP_SIZE_M=1,
                COMBINE_SCALE_LOAD=False, USE_MLS_LOAD=bool(args.triton_mls),
                num_warps=args.triton_waves if args.triton_waves is not None else 4,
                num_stages=args.triton_stages if args.triton_stages is not None else 2,
                use_async_copy=bool(args.triton_async), use_block_pingpong=bool(args.triton_pingpong))


def read_single_tg_stats(root):
    """Use the TG's first-wave -> last-wave interval, with an explicit unit."""
    # PMD also writes m5out/stats.txt for the complete process. Only the
    # device/dispatch files identify individual kernel intervals.
    files = sorted(path for path in root.glob("m5out/*/*/stats.txt")
                   if all(part.isdecimal() for part in path.relative_to(root).parts[1:3]))
    if len(files) != 1:
        raise ValueError(f"expected one dispatch stats.txt, found {len(files)}")
    raw = files[0].read_text()
    block = raw.split("---------- Begin Simulation Statistics ----------")[-1]
    values = {}
    for line in block.splitlines():
        match = re.match(r"^([\w.]+)\s+([-+0-9.eE]+)(?:\s|$)", line)
        if match:
            values[match[1]] = float(match[2])
    wg_counts = [value for key, value in values.items() if re.fullmatch(r"system\.shaders\.CUs\d+\.completedWGs", key)]
    if not wg_counts or any(value < 0 or not value.is_integer() for value in wg_counts) or sum(wg_counts) != 1:
        raise ValueError(f"stats must prove exactly one completed TG; got {wg_counts}")
    start = values["system.shaders.firstWaveStartTick"]
    end = values["system.shaders.lastWaveEndTick"]
    frequency = values["simFreq"]
    if not all(math.isfinite(value) for value in (start, end, frequency)) or not (0 <= start < end and frequency > 0):
        raise ValueError("invalid TG timing interval or tick frequency")
    return {
        "stats_path": str(files[0].relative_to(root)),
        "scope": "first_wave_start_to_last_wave_end_single_tg",
        "first_wave_start_tick": start,
        "last_wave_end_tick": end,
        "elapsed_ticks": end - start,
        "tick_frequency_hz": frequency,
        "time_ns": (end - start) * 1e9 / frequency,
        "completed_tgs": 1,
        "sha256": digest_file(files[0]),
    }


def snapshot_repo(root, destination):
    """Commit + tracked diff + untracked sources reproduce a dirty experiment."""
    root = Path(subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True).strip())
    destination.mkdir(parents=True)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    (destination / "tracked.diff").write_bytes(git("diff", "HEAD", "--binary"))
    names = set(git("diff", "HEAD", "--name-only", "-z").split(b"\0"))
    names.update(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"))
    sources = {}
    for encoded in sorted(names - {b""}):
        name = os.fsdecode(encoded)
        src = root / name
        if {"m5out", "__pycache__", ".pytest_cache"}.intersection(Path(name).parts):
            continue
        if not src.is_file():
            continue
        # Snapshot code/config changes, not transient build or model outputs.
        if src.suffix not in (".py", ".cc", ".cpp", ".h", ".td", ".json", ".mlir", ".txt", ".sh", ".cmake"):
            continue
        dst = destination / "changed_sources" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        sources[name] = digest_file(dst)
    result = {
        "root": str(root),
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "status": git("status", "--short").decode(),
        "diff_sha256": digest_file(destination / "tracked.diff"),
        "changed_source_sha256": sources,
    }
    write_json(destination / "version.json", result)
    return result


def repository_root(path):
    return Path(subprocess.check_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                                        text=True).strip()).resolve()


def verify_sources(manifest):
    """Reject edits between the snapshot and either member of a slow pair."""
    for name in ("triton", "boltops"):
        version = manifest[name]
        root = Path(version["root"])

        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args])

        if (git("rev-parse", "HEAD").decode().strip() != version["commit"]
                or git("status", "--short").decode() != version["status"]
                or hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest() != version["diff_sha256"]):
            raise RuntimeError(f"{name} source changed after the experiment snapshot")
        for source, expected in version["changed_source_sha256"].items():
            path = root / source
            if not path.is_file() or digest_file(path) != expected:
                raise RuntimeError(f"{name} source changed after the experiment snapshot: {source}")
    library = manifest["libtriton"]
    if digest_file(Path(library["path"])) != library["sha256"]:
        raise RuntimeError("libtriton changed after the experiment snapshot")
    for tool in manifest.get("toolchain", {}).values():
        if digest_file(Path(tool["path"])) != tool["sha256"]:
            raise RuntimeError(f"code generation tool changed after the snapshot: {tool['path']}")
    selection = manifest.get("triton_baseline", {}).get("json_config")
    if selection and digest_file(Path(selection["snapshot_path"])) != selection["sha256"]:
        raise RuntimeError("Triton baseline JSON changed after the experiment snapshot")


def save_kernel(compiled):
    # Save the executable and IR before model execution, including failed runs.
    for name, value in compiled.asm.items():
        if isinstance(value, str):
            Path(f"kernel.{name}").write_text(value)
        elif isinstance(value, bytes):
            Path(f"kernel.{name}").write_bytes(value)
    write_json(
        Path("compiled.json"), {
            "kernel_hash": compiled.hash,
            "kernel_metadata": compiled.metadata._asdict(),
            "artifact_sha256": {str(path): digest_file(path)
                                for path in sorted(Path.cwd().glob("kernel.*"))},
        })


def quantize_cpu(x, recipe):
    import torch

    assert x.device.type == "cpu"
    x = x.to(torch.bfloat16).float()
    scale = x.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny) / 448
    return (x / scale).to(torch.float8_e4m3fn), scale


def prepare_cpu(case, recipe, bm, bn, tile_id, seed, gemm="gemm1"):
    import torch

    m, e, topk = case["M"], case["E"], case["topk"]
    if not (0 < m < 2**24 and 0 < topk <= 256):
        raise ValueError("CK token/route encoding requires M < 2**24 and topk <= 256")
    k, out = (case["K"], 2 * case["N"]) if gemm == "gemm1" else (case["N"], case["K"])
    a_rows = m if gemm == "gemm1" else m * topk
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a, sa = quantize_cpu(torch.randn((a_rows, k), generator=gen, device="cpu"), recipe)
    b = torch.empty((e, out, k), dtype=a.dtype, device="cpu")
    sb = torch.empty((e, out, 1), dtype=sa.dtype, device="cpu")
    # Generate full-shaped weights in expert chunks, avoiding a full FP32 copy.
    for expert in range(e):
        b[expert], sb[expert] = quantize_cpu(torch.randn((out, k), generator=gen, device="cpu"), recipe)
    # Synthetic local route distribution, explicitly recorded. E_global remains
    # model metadata; this does not emulate distributed expert communication.
    routing = torch.rand((m, e), generator=gen, device="cpu").topk(topk, dim=1).indices
    flat = routing.flatten()
    sorted_parts, experts, counts = [], [], []
    for expert in range(e):
        routes = (flat == expert).nonzero().flatten().to(torch.int32)
        counts.append(routes.numel())
        if not routes.numel():
            continue
        padding = (-routes.numel()) % bm
        if padding:
            routes = torch.cat((routes, torch.full((padding, ), m * topk, dtype=torch.int32)))
        sorted_parts.append(routes)
        experts.extend([expert] * (routes.numel() // bm))
    sorted_ids = torch.cat(sorted_parts)
    expert_ids = torch.tensor(experts, dtype=torch.int32)
    block_m, block_n = divmod(tile_id, (out + bn - 1) // bn)
    if not 0 <= block_m < expert_ids.numel():
        raise ValueError("selected logical tile is out of range")
    routes = sorted_ids[block_m * bm:(block_m + 1) * bm].long()
    routes = routes[(routes >= 0) & (routes < m * topk)]
    if not routes.numel():
        raise ValueError("performance sample must contain valid routes")
    expert = int(expert_ids[block_m])
    c0, c1 = block_n * bn, min((block_n + 1) * bn, out)
    input_rows = routes // topk if gemm == "gemm1" else routes
    aq = a.float()[input_rows]
    bq = b[expert, c0:c1].float()
    golden = (aq @ bq.T) * sa[input_rows] * sb[expert, c0:c1].T
    weights = torch.rand((m, topk), generator=gen, device="cpu")
    weights = (weights / weights.sum(-1, keepdim=True)).flatten()
    if gemm == "gemm2":
        golden = golden * weights[routes, None]
    golden = golden.to(torch.bfloat16)
    sorted_weights = weights[sorted_ids.long().clamp_max(m * topk - 1)]
    sorted_weights = torch.where(sorted_ids < m * topk, sorted_weights, 0)
    # Match CK's uint32 packing; preserve signed int32 bit patterns on transfer.
    sorted_ids = ((sorted_ids // topk) | ((sorted_ids % topk) << 24)).to(torch.int32)
    tensors = {"A": a, "B": b, "SA": sa, "SB": sb, "Sorted": sorted_ids, "Experts": expert_ids,
               "RouteWeights": weights, "SortedWeights": sorted_weights}
    hashes = {
        name: hashlib.sha256(memoryview(value.contiguous().view(torch.uint8).numpy())).hexdigest()
        for name, value in tensors.items()
    }
    routing_meta = {
        "source": "synthetic_local_topk_without_replacement",
        "sorting_format": "CK: token[23:0], route[31:24], padding token=M",
        "seed": seed,
        "expert_map": "identity_on_local_experts",
        "local_route_counts": counts,
        "logical_tile_id": tile_id,
        "block_m": block_m,
        "block_n": block_n,
        "selected_expert": expert,
        "valid_rows_in_tg": routes.numel(),
        "selected_expert_total_rows": counts[expert],
        "tile_rows": bm,
        "channel_interval": [c0, c1],
        "input_sha256": hashes,
    }
    return tensors, golden, routes, (c0, c1), routing_meta


def worker(args, case):
    import torch
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import gemm_top_w8a8_channel_wasp_persistent

    if os.environ.get("GPU_CHIP") != "sb" or not os.environ.get("PMD_PATH"):
        raise RuntimeError("model experiments require GPU_CHIP=sb and PMD_PATH")
    manifest = json.loads((args.out / "manifest.json").read_text())
    verify_sources(manifest)
    tensors, golden, routes, columns, route_meta = prepare_cpu(case, args.recipe, args.bm, args.bn, args.tile,
                                                               args.seed, args.gemm)
    write_json(Path("inputs.json"), route_meta)
    gpu = {name: value.cuda() for name, value in tensors.items()}
    # CPU initialization followed by H2D; no torch GPU fill/random/reference.
    k, out = (case["K"], 2 * case["N"]) if args.gemm == "gemm1" else (case["N"], case["K"])
    output_rows = tensors["Sorted"].numel()
    output = torch.full((output_rows, out), float("nan"), dtype=golden.dtype, device="cpu").cuda()
    sa, sb = tensors["SA"], tensors["SB"]
    kernel = gemm_top_w8a8_channel_wasp_persistent
    opts = dict(
        BLOCK_SIZE_M=args.bm, BLOCK_SIZE_N=args.bn, BLOCK_SIZE_K=args.bk, A_STAGES=args.a_stages,
        ck_sorting=True,
        USE_ADDR_OFFSET_INT64_A=gpu["A"].numel() * gpu["A"].element_size() >= 2**32 - 1,
        USE_ADDR_OFFSET_INT64_B=gpu["B"].numel() * gpu["B"].element_size() >= 2**32 - 1,
        USE_ADDR_OFFSET_INT64_C=output.numel() * output.element_size() >= 2**32 - 1,
        TOTAL_TILES=args.tile + 1, PARTITION_REGS=tuple(args.regs), PINGPONG=args.pingpong,
        LDS_WRAP=args.lds_wrap, COPY_BYTES=args.copy_bytes, TILE_ID=args.tile,
        num_warps=args.waves, num_stages=2, use_async_copy=True,
        wasp_partition_warps=(args.producer_waves, args.producer_waves, args.waves, args.waves),
        wasp_wdra=args.worker == "wdra",
        amdgpu_enable_max_ilp_scheduling_strategy=False,
    )
    arguments = (gpu["A"], gpu["B"], gpu["SA"], gpu["SB"], output, gpu["Sorted"], gpu["Experts"],
                 case["M"], case["E"], out, k, case["topk"], *sb.stride(), *sa.stride())
    if args.worker == "triton":
        import triton.language as tl
        from boltops.fused_moe.triton.moe_op import fused_moe_kernel

        # Use the library's original kernel, including its normal compiler
        # pipeline. grid=1 selects logical tile zero without shrinking shapes,
        # changing route lists, or shortening the K reduction.
        kernel = fused_moe_kernel
        padded = torch.tensor([tensors["Sorted"].numel()], dtype=torch.int32, device="cpu").cuda()
        a, b, sa_gpu, sb_gpu = (gpu[name] for name in ("A", "B", "SA", "SB"))
        offset_limit = 2**32 - 1
        arguments = (a, b, output, gpu["RouteWeights"], padded, gpu["Experts"], gpu["Sorted"], gpu["SortedWeights"],
                     sa_gpu, sb_gpu, b, out, k, tensors["Sorted"].numel(), case["M"] * case["topk"], *a.stride(),
                     b.stride(0), b.stride(2), b.stride(1), *output.stride(), *sa.stride(), sb.stride(0), sb.stride(2),
                     sb.stride(1), 0, 0, a.shape[0])
        opts = dict(
            group_n=0, group_k=0,
            MUL_ROUTED_WEIGHT=args.gemm == "gemm2", USE_ADDR_OFFSET_INT64_A=a.numel() * a.element_size()
            >= offset_limit, USE_ADDR_OFFSET_INT64_B=b.numel() * b.element_size() >= offset_limit,
            USE_ADDR_OFFSET_INT64_C=output.numel() * output.element_size() >= offset_limit,
            top_k=case["topk"] if args.gemm == "gemm1" else 1,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=args.recipe == "fp8_channel", use_int8_w8a8=False, use_int8_w8a16=False,
            per_channel_quant=args.recipe == "fp8_channel", c_sorted=True, a_sorted=False, bottom_a_use_mls_load=False,
            ck_sorting=True, ck_topk=case["topk"], naive_block_assignment=False, NUM_XCDS=1,
            SCALE_BIAS_WITH_ROUTED_WEIGHT=False, ADD_BIAS=False)
        # Every JSON setting reaches the original kernel/backend unchanged.
        # Workers use the frozen manifest, never reconstruct a tuned baseline.
        opts.update(manifest["triton_baseline"]["options"])
    write_json(
        Path("launch.json"), {
            "grid": [1],
            "options": opts,
            "shape": {key: case[key]
                      for key in ("M", "E", "N", "K", "topk")},
            "tensors": {
                name: {
                    "shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype), "device":
                    str(value.device), "address": hex(value.data_ptr()), "bytes": value.numel() * value.element_size()
                }
                for name, value in {**gpu, "C": output}.items()
            },
        })
    # JIT warmup only compiles; it does not launch a GPU kernel.
    compiled = (kernel.run(*arguments, grid=(1, ), warmup=True, **opts) if args.worker == "triton" else kernel.warmup(
        *arguments, grid=(1, ), **opts))
    save_kernel(compiled)
    if compiled.metadata.shared > 128 * 1024:
        raise RuntimeError(f"gfx946 LDS capacity exceeded: {compiled.metadata.shared} > 131072 bytes")
    kernel[(1, )](*arguments, **opts)
    torch.cuda.synchronize()
    actual = output.cpu()
    c0, c1 = columns
    block_m = route_meta["block_m"]
    tile_routes = tensors["Sorted"][block_m * args.bm:(block_m + 1) * args.bm]
    valid_rows = (tile_routes & 0x00FFFFFF) < case["M"]
    result_rows = block_m * args.bm + torch.arange(args.bm)[valid_rows]
    torch.save({"actual": actual[result_rows, c0:c1], "golden": golden, "routes": routes, "columns": columns},
               "numerical.pt")
    observed = actual[result_rows, c0:c1].float()
    expected = golden.float()
    difference = (observed - expected).abs()
    close = torch.isclose(observed, expected, rtol=0.02, atol=0.002)
    max_absolute_error = float(difference.max())
    write_json(
        Path("validation.json"), {
            "rtol": 0.02,
            "atol": 0.002,
            "checked_elements": golden.numel(),
            "mismatched_elements": int((~close).sum()),
            "max_absolute_error": max_absolute_error if math.isfinite(max_absolute_error) else None,
            "nonfinite_elements": int((~torch.isfinite(observed)).sum()),
        })
    # FP32 accumulation followed by one output rounding step. This is a finite random-input check,
    # not a saturation, extreme exponent or quantizer-boundary qualification.
    torch.testing.assert_close(actual[result_rows, c0:c1], golden, rtol=0.02, atol=0.002)
    written = int(torch.isfinite(actual).sum())
    written_elements = args.bm * (c1 - c0) if args.worker != "triton" else golden.numel()
    if written != written_elements:
        raise AssertionError(f"unexpected output writes: {written} versus {written_elements}")
    unselected = torch.ones(actual.shape, dtype=torch.bool, device="cpu")
    if args.worker != "triton":
        unselected[block_m * args.bm:(block_m + 1) * args.bm, c0:c1] = False
        padding = actual[block_m * args.bm:(block_m + 1) * args.bm, c0:c1][~valid_rows]
        if not (padding == 0).all():
            raise AssertionError("sorted padding rows must be zero")
    else:
        unselected[result_rows, c0:c1] = False
    if not torch.isnan(actual[unselected]).all():
        raise AssertionError("a location outside the selected TG output lost its NaN sentinel")
    verify_sources(manifest)
    loaded_model = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "libgem5" in line and "/" in line:
            path = Path(line.split()[-1])
            if path.is_file() and str(path) not in loaded_model:
                loaded_model[str(path)] = digest_file(path)
    result = {
        "status": "passed",
        "mode": args.worker,
        "grid": [1],
        "route": route_meta,
        "numerical": {
            "rtol": 0.02, "atol": 0.002, "checked_elements": golden.numel(), "output_dtype": str(golden.dtype), "guard":
            "all_unselected_output_remains_nan"
        },
        "kernel_metadata": compiled.metadata._asdict(),
        "kernel_hash": compiled.hash,
        "artifact_sha256": {str(path): digest_file(path)
                            for path in sorted(Path.cwd().glob("kernel.*"))},
        "loaded_model_sha256": loaded_model,
        "timing": read_single_tg_stats(Path.cwd()),
    }
    write_json(Path("result.json"), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cases = json.loads(CASES.read_text())["cases"]
    parser.add_argument("--case", choices=[case["id"] for case in cases], default="qwen35-small-m64")
    parser.add_argument("--recipe", choices=["fp8_channel"], required=True)
    parser.add_argument("--gemm", choices=["gemm1", "gemm2"], default="gemm1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bm", type=int, help="default: 128")
    parser.add_argument("--bn", type=int, help="default: 256")
    parser.add_argument("--bk", type=int, help="default: 128")
    parser.add_argument("--stages", type=int, help="B pipeline depth; fixed at 3")
    parser.add_argument("--a-stages", type=int, choices=[2], default=2,
                        help="A pipeline depth (fixed at 2 with B3)")
    parser.add_argument("--regs", type=int, nargs="+")
    parser.add_argument("--waves", type=int, choices=[1, 2, 4], help="waves per partition; WDRA requires 4")
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--gc-ghz", type=float, default=1.8, help="model GC frequency; default: 1.8 GHz")
    parser.add_argument("--include-wdra", action="store_true")
    parser.add_argument("--include-triton", action="store_true",
                        help="also measure the original ordinary Triton MoE kernel")
    parser.add_argument("--modes", nargs="+", choices=["triton", "wasp", "wdra"],
                        help="explicit subset for independent baseline/configuration searches")
    parser.add_argument("--triton-config-json", type=Path,
                        help="ordinary FP8 channel Triton JSON; nearest M, all settings preserved")
    parser.add_argument("--triton-config-source", help="source URL and commit for the baseline JSON")
    parser.add_argument("--manual-triton-config", action="store_true",
                        help="explicitly opt into a manual baseline search, not JSON qualification")
    parser.add_argument("--triton-stages", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--triton-bk", type=int, choices=[64, 128, 256, 512],
                        help="baseline K tile; defaults to Gluon's BK, with the same complete reduction")
    parser.add_argument("--triton-waves", type=int, choices=[2, 4, 8, 16])
    parser.add_argument("--triton-async", action="store_true", default=None)
    parser.add_argument("--triton-mls", action="store_true", default=None)
    parser.add_argument("--triton-pingpong", action="store_true", default=None)
    parser.add_argument("--producer-waves", type=int, choices=[1, 2, 4], help="wave count shared by both producers")
    parser.add_argument("--lds-wrap", type=int, choices=[0, 16, 32, 64], help="default: 32")
    parser.add_argument("--copy-bytes", type=int, choices=[4, 8, 16], default=16)
    parser.add_argument("--worker-timeout", type=float, default=300,
                        help="maximum wall seconds per isolated model worker; failed runs retain artifacts")
    parser.add_argument("--pingpong", action=argparse.BooleanOptionalAction, default=True,
                        help="alternate consumer partitions with directional ebarriers (default: enabled)")
    parser.add_argument("--worker", choices=["triton", "wasp", "wdra"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    modes = args.modes or (["triton"] if args.include_triton else []) + (["wasp", "wdra"] if args.include_wdra else ["wdra"])
    args.channel_n_split = any(mode != "triton" for mode in modes)
    args.consumers = 2 if args.channel_n_split else 1
    if not math.isfinite(args.gc_ghz) or args.gc_ghz <= 0:
        parser.error("gc-ghz must be finite and positive")
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        parser.error("worker-timeout must be finite and positive")
    case = next(case for case in cases if case["id"] == args.case)
    selection = None
    # Child processes consume the already frozen manifest and geometry.
    if not args.worker and args.triton_config_json:
        if args.manual_triton_config:
            parser.error("choose JSON or manual Triton configuration, not both")
        try:
            selection = select_triton_config(args.triton_config_json, case["M"], recipe=args.recipe)
            baseline_options(args, selection)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    for name, fallback in (("bm", 128), ("bn", 256), ("bk", 128), ("stages", 3),
                           ("waves", 4)):
        if getattr(args, name) is None:
            setattr(args, name, fallback)
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import gemm1_partition_regs
    # Measured channel defaults; explicit command-line choices remain authoritative.
    if args.regs is None:
        input_k = case["K"] if args.gemm == "gemm1" else case["N"]
        wide_a = case["M"] * input_k >= 2**32 - 1
        args.regs = list(gemm1_partition_regs(wide_a))
    if args.lds_wrap is None:
        args.lds_wrap = 32
    if args.producer_waves is None:
        args.producer_waves = args.waves
    if args.channel_n_split:
        if args.bk != 128:
            parser.error("channel Gluon requires BK128")
        if (args.bm, args.stages) != (128, 3) or args.bn not in (128, 256):
            parser.error("channel Gluon requires BM128, BN128/256 and B STAGES=3")
        if len(args.regs) != 4:
            parser.error("regs requires two producer quotas and two consumer quotas")
        if args.waves != 4 or args.producer_waves != 4:
            parser.error("channel Gluon uses four waves in each producer and consumer partition")
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import gemm1_mma_warps, gemm1_mls_warps, INST_N
    args.mls_tile = (64, 32)
    args.mls_warps = gemm1_mls_warps((args.bk, args.bn // args.consumers), args.mls_tile, args.producer_waves)
    args.mma_tiles_n = args.mls_tile[1] // INST_N.value
    args.mma_warps = list(gemm1_mma_warps(args.bm, args.bn // args.consumers,
                                       args.waves, args.mma_tiles_n))
    if any(mode != "triton" for mode in ([args.worker] if args.worker else modes)):
        if args.recipe != "fp8_channel" or args.gemm != "gemm1":
            parser.error("the Gluon path currently supports only FP8 channel GEMM1")
    if not args.worker and "triton" in modes and selection is None and not args.manual_triton_config:
        parser.error(
            "ordinary Triton comparisons require --triton-config-json; use --manual-triton-config only for explicit searches"
        )
    if "triton" in modes and args.tile != 0:
        parser.error("the unchanged ordinary Triton kernel currently samples logical tile zero only")
    if "wdra" in modes and args.waves != 4:
        parser.error("WDRA requires four waves per partition")
    if "wdra" in modes and args.producer_waves != 4:
        parser.error("WDRA requires four producer waves")
    if args.bm not in (16, 32, 64, 128) or args.bn not in (32, 64, 128, 256):
        parser.error("GEMM candidates require BM=16/32/64/128, BN=32/64/128/256")
    reduction = case["K"] if args.gemm == "gemm1" else case["N"]
    if args.bk not in (64, 128, 256, 512) or reduction % args.bk or not 1 <= args.stages <= 8 or args.tile < 0:
        parser.error("invalid BK, STAGES or logical tile")
    if args.worker:
        model_args = ast.literal_eval(os.environ.get("GPU_ARGS", "[]"))
        if not isinstance(model_args, list) or not all(isinstance(arg, str) for arg in model_args):
            parser.error("GPU_ARGS must be a list of strings")
        os.environ["GPU_ARGS"] = str([arg for arg in model_args if not arg.startswith("--gpu-clock=")]
                                    + [f"--gpu-clock={args.gc_ghz:g}GHz"])
        worker(args, case)
        return
    import triton
    import torch
    from triton._C import libtriton
    from triton.backends.hcu.compiler import HIPBackend
    args.out = args.out.resolve()
    triton_root = repository_root(Path(triton.__file__).resolve().parent)
    boltops_root = repository_root(Path(__file__).resolve().parent)
    if any(args.out.is_relative_to(root) for root in (triton_root, boltops_root)):
        parser.error("experiment output must be outside both source worktrees")
    args.out.mkdir(parents=True, exist_ok=False)
    triton_config = baseline_options(args, selection)
    if selection:
        snapshot = args.out / "triton-baseline.json"
        shutil.copyfile(args.triton_config_json, snapshot)
        selection["snapshot_path"] = str(snapshot)
        selection["source"] = args.triton_config_source
        selection["recipe_scope"] = "fp8_channel"
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "torch": {"version": torch.__version__, "cpu_threads": torch.get_num_threads()},
        "case": case,
        "gemm": args.gemm,
        "recipe": args.recipe,
        "modes": modes,
        "triton_baseline": {
            "kernel": "boltops.fused_moe.triton.moe_op.fused_moe_kernel", "stages": triton_config["num_stages"],
            "waves": triton_config["num_warps"], "bm": triton_config["BLOCK_SIZE_M"], "bn":
            triton_config["BLOCK_SIZE_N"], "bk": triton_config["BLOCK_SIZE_K"], "block_pingpong":
            triton_config.get("use_block_pingpong", False), "async_copy": triton_config.get("use_async_copy", False),
            "matrix_load_b": triton_config.get("USE_MLS_LOAD", False), "configuration_kind":
            "json" if selection else "manual", "json_config": selection, "options": triton_config
        },
        "worker_timeout_seconds": args.worker_timeout,
        "gc_ghz": args.gc_ghz,
        "config": {
            name: getattr(args, name)
            for name in ("bm", "bn", "bk", "stages", "a_stages", "regs", "mma_warps", "waves", "tile", "seed",
                         "mma_tiles_n", "mls_tile", "mls_warps", "consumers", "producer_waves", "lds_wrap", "copy_bytes",
                         "pingpong", "channel_n_split")
        },
        "environment": {
            key: os.environ.get(key)
            for key in ("GPU_CHIP", "GPU_DFLAGS", "PMD_PATH", "PYTHONPATH", "TRITON_CACHE_DIR", "TRITON_HIP_CLANG_PATH",
                        "TRITON_HIP_LLD_PATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "triton": snapshot_repo(triton_root, args.out / "sources/triton"),
        "boltops": snapshot_repo(boltops_root, args.out / "sources/boltops"),
        "libtriton": {"path": libtriton.__file__, "sha256": digest_file(Path(libtriton.__file__))},
        "toolchain": {
            name: {
                "path": str(path), "sha256": digest_file(path), "version":
                subprocess.check_output([str(path), "--version"], text=True)
            }
            for name, path in (("clang", HIPBackend.path_to_rocm_clang()), )
        },
    }
    pmd = Path(os.environ.get("PMD_PATH", "/opt/rocm/pmd"))
    manifest["pmd"] = {"root": str(pmd)}
    readme = pmd / "readme.txt"
    if readme.is_file():
        shutil.copyfile(readme, args.out / "pmd-readme.txt")
        manifest["pmd"]["readme_sha256"] = digest_file(readme)
    for pattern in ("gem5.opt", "libgem5_opt.so"):
        for binary in sorted(pmd.rglob(pattern)):
            if binary.is_file():
                manifest["pmd"][str(binary.relative_to(pmd))] = digest_file(binary)
    shutil.copyfile(CASES, args.out / "qwen_cases.json")
    write_json(args.out / "manifest.json", manifest)
    results = {}
    for mode in modes:
        directory = args.out / mode
        directory.mkdir()
        command = [
            sys.executable, "-m", "boltops.tools.gluon_moe_gemm1", "--worker", mode, "--case", args.case, "--recipe",
            args.recipe, "--gemm", args.gemm, "--out",
            str(args.out), "--bm",
            str(args.bm), "--bn",
            str(args.bn), "--bk",
            str(args.bk), "--stages",
            str(args.stages), "--a-stages", str(args.a_stages), "--tile",
            str(args.tile), "--seed",
            str(args.seed), "--regs", *map(str, args.regs), "--waves",
            str(args.waves), "--modes", *modes
        ]
        command.extend(["--gc-ghz", str(args.gc_ghz)])
        command.extend(["--producer-waves", str(args.producer_waves),
                        "--lds-wrap", str(args.lds_wrap), "--copy-bytes", str(args.copy_bytes)])
        if args.triton_async:
            command.append("--triton-async")
        if args.triton_mls:
            command.append("--triton-mls")
        if args.triton_pingpong:
            command.append("--triton-pingpong")
        command.append("--pingpong" if args.pingpong else "--no-pingpong")
        write_json(directory / "command.json", command)
        timed_out = False
        with (directory / "run.log").open("w") as log:
            try:
                completed = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=args.worker_timeout)
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                timed_out = True
        result_path = directory / "result.json"
        if returncode == 0 and result_path.is_file():
            results[mode] = json.loads(result_path.read_text())
        else:
            results[mode] = {
                "status": "failed", "exit_code": returncode, "log": f"{mode}/run.log", "reason":
                "worker_timeout" if timed_out else "worker_failed"
            }
        write_json(args.out / "summary.json", results)
    if all(result["status"] == "passed" for result in results.values()):
        hashes = [result["route"]["input_sha256"] for result in results.values()]
        if any(value != hashes[0] for value in hashes[1:]):
            raise RuntimeError("paired modes generated different inputs; results cannot be compared")
        if "triton" in results:
            baseline = results["triton"]["timing"]["time_ns"]
            for mode in modes:
                results[mode]["speedup_vs_triton"] = baseline / results[mode]["timing"]["time_ns"]
        write_json(args.out / "summary.json", results)
    else:
        raise SystemExit("one or more modes failed; no performance ranking was produced")


if __name__ == "__main__":
    main()
