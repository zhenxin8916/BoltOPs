# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Compare BM64/BN512 Gluon GEMM2 against ordinary Triton on identical inputs.

RUN_GLUON_GEMM2_BM64_BN512_EQUIVALENCE=1 enables the gfx946 test. Fixed-seed routes and
quantized inputs are generated once, shared by both backends, and hashed.
Multiple M blocks and two N tiles exercise A reuse and metadata generations.
"""
import hashlib
import json
import os

import pytest
import torch


@pytest.mark.skipif(os.environ.get("RUN_GLUON_GEMM2_BM64_BN512_EQUIVALENCE") != "1",
                    reason="opt-in GPU equivalence")
@pytest.mark.parametrize("kind,out,mul,wide", [
    ("fp8", torch.bfloat16, True, False),
    ("int8", torch.bfloat16, True, False),
    ("fp8", torch.float16, False, True),
    ("int8", torch.float16, False, True),
    ("fp8_e5m2", torch.bfloat16, True, False),
    ("fp8_mixed", torch.float16, True, True),
])
def test_gluon_gemm2_matches_triton(monkeypatch, record_property, kind, out, mul, wide):
    import triton.language as tl
    from boltops.tools.gluon_moe_gemm1 import prepare_cpu
    from boltops.fused_moe.gluon import (
        gemm_bottom_w8a8_channel_wasp_persistent_bm64xbn512xbk64 as mod,
    )
    from boltops.fused_moe.triton import moe_op

    torch.manual_seed(3101)
    t, *_ = prepare_cpu(dict(M=193, E=2, N=256, K=1024, topk=2),
                       "fp8_channel", 64, 256, 0, 3101, gemm="gemm2")
    ids = t["Sorted"].long()
    routes = (ids & 0xffffff) * 2 + ((ids >> 24) & 255)
    valid = routes < 386
    gather = routes.clamp_max(385)
    if kind == "int8":
        t["A"] = torch.randint(-16, 17, t["A"].shape, dtype=torch.int8)
        t["B"] = torch.randint(-16, 17, t["B"].shape, dtype=torch.int8)
        t["SA"] = torch.full_like(t["SA"], .01)
        t["SB"] = torch.full_like(t["SB"], .02)
    if kind == "fp8_e5m2":
        t["A"] = t["A"].float().to(torch.float8_e5m2)
        t["B"] = t["B"].float().to(torch.float8_e5m2)
    elif kind == "fp8_mixed":
        t["B"] = t["B"].float().to(torch.float8_e5m2)
    a = t["A"].view(torch.uint8)[gather].clone()
    a[~valid] = 0
    t["A"] = a.view(t["A"].dtype)
    t["SA"] = t["SA"][gather].clone()
    t["SA"][~valid] = 0
    rows = routes.numel()
    golden = torch.zeros((386, 1024), dtype=out)
    for m in range(rows // 64):
        span = slice(m * 64, (m + 1) * 64)
        mask = valid[span]
        rt = routes[span][mask]
        expert = int(t["Experts"][m])
        value = ((t["A"][span].float()[mask] @ t["B"][expert].float().T)
                 * t["SA"][span][mask] * t["SB"][expert, :, 0][None, :])
        if mul:
            value = value * t["RouteWeights"][rt, None]
        golden[rt] = value.to(out)
    t["WorkCount"] = torch.tensor([rows], dtype=torch.int32)
    gpu = {name: value.cuda() for name, value in t.items()}
    if wide:
        # Keep non-unit channel strides through the actual device launch.
        for name in ("SA", "SB"):
            value = gpu[name]
            backing = torch.empty((*value.shape[:-1], value.shape[-1] * 2),
                                  dtype=value.dtype, device=value.device)
            view = backing[..., ::2]
            view.copy_(value)
            gpu[name] = view
    g = torch.zeros_like(golden).cuda()
    r = torch.zeros_like(g)
    opts = dict(mod.DEFAULT_LAUNCH_OPTIONS, N=1024, K=256, top_k=2,
                MUL_ROUTED_WEIGHT=mul, USE_ADDR_OFFSET_INT64_A=wide,
                USE_ADDR_OFFSET_INT64_B=wide, USE_ADDR_OFFSET_INT64_C=wide,
                stride_asm=gpu["SA"].stride(0), stride_bse=gpu["SB"].stride(0),
                stride_bsn=gpu["SB"].stride(1),
                num_tokens_post_padded_ptr=gpu["WorkCount"])
    mod.gemm_bottom_w8a8_channel_wasp_persistent[(1,)](
        gpu["A"], gpu["B"], gpu["SA"], gpu["SB"], g,
        gpu["SortedWeights"] if mul else None, gpu["Sorted"], gpu["Experts"],
        rows, 386, rows // 64 * 2, **opts)

    original = moe_op.fused_moe_kernel
    calls = []

    class Track:
        def __getitem__(self, grid):
            launch = original[grid]

            def run(*args, **kwargs):
                calls.append(True)
                return launch(*args, **kwargs)

            return run

    # Explicitly disable Gluon dispatch: this must exercise ordinary Triton.
    monkeypatch.setattr(moe_op, "fused_moe_kernel", Track())
    monkeypatch.setattr(moe_op, "_try_gluon_channel_gemm2", lambda *a, **kw: False)
    monkeypatch.setattr(moe_op, "_try_gluon_channel_gemm1", lambda *a, **kw: False)
    monkeypatch.setattr(moe_op, "_USE_MOE_PERSISTENT_KERNEL", False)
    monkeypatch.setattr(moe_op, "splitk_size", 0)
    moe_op.fused_moe(
        gpu["A"], gpu["B"], r, gpu["SA"], gpu["SB"], None,
        gpu["RouteWeights"].reshape(193, 2),
        torch.zeros((193, 2), dtype=torch.int32, device="cuda"),
        gpu["Sorted"], gpu["SortedWeights"], gpu["Experts"], gpu["WorkCount"],
        mul, 1, tl.bfloat16 if out == torch.bfloat16 else tl.float16,
        use_fp8_w8a8=kind.startswith("fp8"), use_int8_w8a8=kind == "int8",
        per_channel_quant=True, c_sorted=False, a_sorted=True, ck_sorting=True,
        ck_topk=2, config=dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                              GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False,
                              USE_MLS_LOAD=False, num_warps=4, num_stages=2))
    gh, rh = g.cpu(), r.cpu()
    assert len(calls) == 1
    torch.testing.assert_close(gh, golden, rtol=.02, atol=.002)
    torch.testing.assert_close(rh, golden, rtol=.02, atol=.002)
    torch.testing.assert_close(gh, rh, rtol=0, atol=0)
    hashes = {name: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
              for name, value in t.items()}
    report = dict(input_sha256=hashes, kind=kind, output=str(out),
                  mul_routed_weight=mul, forced_int64=wide, elements=g.numel(),
                  differences=int((gh != rh).sum()))
    record_property("equivalence", json.dumps(report))
    print(json.dumps(report), flush=True)


@pytest.mark.parametrize("recipe,dtype", [
    ("fp8_channel", torch.float8_e4m3fn),
    ("fp8_channel", torch.float8_e5m2),
    ("int8_channel", torch.int8),
])
def test_bm64_bn512_support_query(recipe, dtype):
    """Metadata selection follows the sorted-row and fixed K256 contract."""
    from boltops.fused_moe.gluon import (
        gemm_bottom_w8a8_channel_wasp_persistent_bm64xbn512xbk64 as mod,
    )
    a = torch.empty((128, 256), dtype=dtype)
    b = torch.empty((2, 512, 256), dtype=dtype)
    sa = torch.empty((128, 1), dtype=torch.float32)
    sb = torch.empty((2, 512, 1), dtype=torch.float32)
    options = dict(arch="gfx946", cu_count=48, top_k=8,
                   config={"BLOCK_SIZE_M": 64}, scale_recipe=recipe)
    assert mod.is_persistent_channel_gemm2_supported(a, b, sa, sb, **options)
    assert not mod.is_persistent_channel_gemm2_supported(
        a, b, sa, sb, **dict(options, config={"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 2}))
    assert not mod.is_persistent_channel_gemm2_supported(a, b, sa, sb, **options, ck_sorting=False)
    assert not mod.is_persistent_channel_gemm2_supported(a, b, sa, sb, **options, c_sorted=True)
