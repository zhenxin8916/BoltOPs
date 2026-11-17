# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Compare BM32/BN512 Gluon GEMM2 against ordinary Triton on identical inputs.

RUN_GLUON_GEMM2_BM32_BN512_EQUIVALENCE=1 enables the gfx946 test. Fixed-seed routes and
quantized inputs are generated once, shared by both backends, and hashed.
Multiple M blocks and two N tiles exercise A reuse and metadata generations.
"""
import hashlib
import json
import os

import pytest
import torch


@pytest.mark.skipif(os.environ.get("RUN_GLUON_GEMM2_BM32_BN512_EQUIVALENCE") != "1",
                    reason="opt-in GPU equivalence")
@pytest.mark.parametrize("kind,out,mul,wide", [
    ("fp8", torch.bfloat16, True, False),
    ("int8", torch.bfloat16, True, False),
    ("fp8", torch.float16, False, True),
    ("int8", torch.float16, False, True),
    ("fp8_e5m2", torch.bfloat16, True, False),
    ("fp8_mixed", torch.float16, True, True),
    ("fp8_capped", torch.bfloat16, True, False),
    ("fp8", torch.bfloat16, True, "a"),
    ("fp8", torch.bfloat16, True, "b"),
    ("fp8", torch.bfloat16, True, "c"),
    ("fp8_mixed_reverse", torch.float16, True, True),
    ("fp8_zero_work", torch.bfloat16, True, False),
])
def test_gluon_gemm2_matches_triton(monkeypatch, record_property, kind, out, mul, wide):
    import triton.language as tl
    from boltops.tools.gluon_moe_gemm1 import prepare_cpu
    from boltops.fused_moe.gluon import (
        gemm_bottom_w8a8_channel_wasp_persistent_bm32xbn512xbk64 as mod,
    )
    from boltops.fused_moe.triton import moe_op

    torch.manual_seed(3101)
    t, *_ = prepare_cpu(dict(M=193, E=2, N=256, K=1024, topk=2),
                       "fp8_channel", 32, 256, 0, 3101, gemm="gemm2")
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
    elif kind == "fp8_mixed_reverse":
        t["A"] = t["A"].float().to(torch.float8_e5m2)
    a = t["A"].view(torch.uint8)[gather].clone()
    a[~valid] = 0
    t["A"] = a.view(t["A"].dtype)
    t["SA"] = t["SA"][gather].clone()
    t["SA"][~valid] = 0
    rows = routes.numel()
    # Keep allocated capacity/TOTAL_TILES larger than the device work limit.
    capped = kind == "fp8_capped"
    zero_work = kind == "fp8_zero_work"
    active_rows = 0 if zero_work else rows - 32 if capped else rows
    preserve_inactive = capped or zero_work
    sentinel = -11 if preserve_inactive else 0
    golden = torch.full((386, 1024), sentinel, dtype=out)
    active_output = torch.zeros(386, dtype=torch.bool)
    for m in range(active_rows // 32):
        span = slice(m * 32, (m + 1) * 32)
        mask = valid[span]
        rt = routes[span][mask]
        expert = int(t["Experts"][m])
        value = ((t["A"][span].float()[mask] @ t["B"][expert].float().T)
                 * t["SA"][span][mask] * t["SB"][expert, :, 0][None, :])
        if mul:
            value = value * t["RouteWeights"][rt, None]
        golden[rt] = value.to(out)
        active_output[rt] = True
    t["WorkCount"] = torch.tensor([active_rows], dtype=torch.int32)
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
    g = torch.full_like(golden, sentinel).cuda()
    r = torch.full_like(g, sentinel)
    # A/B/C wide address decisions are independent booleans, never string constexprs.
    wide_a, wide_b, wide_c = (wide is True or wide == name for name in ("a", "b", "c"))
    opts = dict(mod.DEFAULT_LAUNCH_OPTIONS, BLOCK_SIZE_M=32, N=1024, K=256, top_k=2,
                MUL_ROUTED_WEIGHT=mul, USE_ADDR_OFFSET_INT64_A=wide_a,
                USE_ADDR_OFFSET_INT64_B=wide_b, USE_ADDR_OFFSET_INT64_C=wide_c,
                stride_asm=gpu["SA"].stride(0), stride_bse=gpu["SB"].stride(0),
                stride_bsn=gpu["SB"].stride(1),
                num_tokens_post_padded_ptr=gpu["WorkCount"])
    if kind == "fp8" and not wide:
        # Exercise the real framework bridge and its device row-count cap.
        from types import SimpleNamespace
        with monkeypatch.context() as dispatch_patch:
            dispatch_patch.setattr(moe_op.arch_info, "get_device_info",
                                   lambda *_: SimpleNamespace(arch="gfx946", cu_count=1))
            assert moe_op._try_gluon_channel_gemm2(
                gpu["A"], gpu["B"], g, gpu["SA"], gpu["SB"],
                gpu["SortedWeights"], gpu["Sorted"], gpu["Experts"],
                gpu["WorkCount"], 2, {"BLOCK_SIZE_M": 32},
                use_fp8_w8a8=True, use_int8_w8a8=False, per_channel_quant=True,
                block_shape=None, c_sorted=False, a_sorted=True, ck_sorting=True,
                ck_topk=2, naive_block_assignment=False, mul_routed_weight=mul,
                B_bias=None, B_zp=None)
    else:
        mod.gemm_bottom_w8a8_channel_wasp_persistent[(1,)](
            gpu["A"], gpu["B"], gpu["SA"], gpu["SB"], g,
            gpu["SortedWeights"] if mul else None, gpu["Sorted"], gpu["Experts"],
            rows, 386, rows // 32 * 2, **opts)

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
        ck_topk=2, config=dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                              GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False,
                              USE_MLS_LOAD=False, num_warps=4, num_stages=2))
    gh, rh = g.cpu(), r.cpu()
    assert len(calls) == 1
    if preserve_inactive:
        assert active_rows < rows
        assert torch.all(gh[~active_output] == sentinel)
        assert torch.all(rh[~active_output] == sentinel)
    torch.testing.assert_close(gh, golden, rtol=.02, atol=.002)
    torch.testing.assert_close(rh, golden, rtol=.02, atol=.002)
    torch.testing.assert_close(gh, rh, rtol=0, atol=0)
    hashes = {name: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
              for name, value in t.items()}
    report = dict(input_sha256=hashes, kind=kind, output=str(out),
                  mul_routed_weight=mul, forced_int64=wide, elements=g.numel(),
                  differences=int((gh != rh).sum()), sorted_row_capacity=rows,
                  device_work_count=active_rows, inactive_output_rows=int((~active_output).sum()),
                  inactive_sentinel=sentinel if preserve_inactive else None,
                  address_int64=dict(A=wide_a, B=wide_b, C=wide_c))
    record_property("equivalence", json.dumps(report))
    print(json.dumps(report), flush=True)


@pytest.mark.parametrize("recipe,dtype", [
    ("fp8_channel", torch.float8_e4m3fn),
    ("fp8_channel", torch.float8_e5m2),
    ("int8_channel", torch.int8),
])
def test_bm32_bn512_support_query(recipe, dtype):
    """Metadata selection follows the sorted-row and fixed K256 contract."""
    from boltops.fused_moe.gluon import (
        gemm_bottom_w8a8_channel_wasp_persistent_bm32xbn512xbk64 as mod,
    )
    a = torch.empty((128, 256), dtype=dtype)
    b = torch.empty((2, 512, 256), dtype=dtype)
    sa = torch.empty((128, 1), dtype=torch.float32)
    sb = torch.empty((2, 512, 1), dtype=torch.float32)
    options = dict(arch="gfx946", cu_count=48, top_k=8,
                   config={"BLOCK_SIZE_M": 32}, scale_recipe=recipe)
    assert mod.is_persistent_channel_gemm2_supported(a, b, sa, sb, **options)
    assert not mod.is_persistent_channel_gemm2_supported(
        a, b, sa, sb, **dict(options, config={"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 2}))
    assert not mod.is_persistent_channel_gemm2_supported(a, b, sa, sb, **options, ck_sorting=False)
    assert not mod.is_persistent_channel_gemm2_supported(a, b, sa, sb, **options, c_sorted=True)
