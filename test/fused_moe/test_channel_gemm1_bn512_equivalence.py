# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Opt-in BN512 CK-sorted tests with fixed inputs shared by both backends."""
import os
import json
import pytest
import torch

@pytest.mark.parametrize("kind,output_kind,wide", [
    ("fp8", "bf16", False), ("int8", "bf16", False),
    ("fp8", "fp16", True), ("int8", "fp16", True),
    ("fp8e5", "bf16", False), ("fp8mixed", "bf16", False),
])
def test_bn512_matches_ordinary_triton(monkeypatch, kind, output_kind, wide, record_property):
    if os.environ.get("RUN_GLUON_GEMM1_BN512_EQUIVALENCE") != "1":
        pytest.skip("Set RUN_GLUON_GEMM1_BN512_EQUIVALENCE=1 on HCU")
    import triton.language as tl
    from boltops.fused_moe.gluon import gemm_top_w8a8_channel_wasp_persistent_bm128xbn512xbk128 as gluon
    from boltops.fused_moe.triton import moe_op
    rng = torch.Generator().manual_seed(3197)
    m, e, n, k, topk = 97, 2, 1024, 512, 2
    blocks = [torch.cat((torch.randperm(m * topk, generator=rng, dtype=torch.int32)[:120],
                         torch.full((8,), m * topk, dtype=torch.int32))) for _ in range(3)]
    flat = torch.cat(blocks)
    valid = flat < m * topk
    packed = torch.where(valid, (flat // topk) | ((flat % topk) << 24), m).int()
    experts = torch.tensor([1, 0, -1], dtype=torch.int32)
    dtype = torch.int8 if kind == "int8" else (torch.float8_e5m2 if kind == "fp8e5" else torch.float8_e4m3fn)
    group_size_m = 2 if kind == "fp8e5" else 1
    out = torch.bfloat16 if output_kind == "bf16" else torch.float16
    a = torch.randint(-8, 9, (m, k), generator=rng).to(dtype)
    b = torch.randint(-8, 9, (e, n, k), generator=rng).to(torch.float8_e5m2 if kind == "fp8mixed" else dtype)
    # Non-unit scale strides exercise independent scale-address calculations.
    sa = torch.full((m, 3), .125)[:, 1:2]
    sb = torch.full((e, n, 3), .0625)[:, :, 1:2]
    golden = torch.zeros((384, n), dtype=out)
    for q in range(3):
        mask = valid[q * 128:(q + 1) * 128]
        token = (flat[q * 128:(q + 1) * 128] // topk)[mask]
        if experts[q] >= 0:
            value = (a.float()[token] @ b[int(experts[q])].float().T) * sa[token] * sb[int(experts[q]), :, 0][None, :]
            golden[q * 128:(q + 1) * 128][mask] = value.to(out)
    gpu = {name: value.cuda() for name, value in dict(A=a, B=b, SA=sa, SB=sb, Sorted=packed,
        Experts=experts, Count=torch.tensor([384], dtype=torch.int32)).items()}
    a, b, sa, sb, ids, experts, count = (gpu[name] for name in ("A", "B", "SA", "SB", "Sorted", "Experts", "Count"))
    actual = torch.full_like(golden, float("nan"), device="cuda")
    ordinary = torch.zeros_like(actual)
    options = dict(gluon.DEFAULT_LAUNCH_OPTIONS, BLOCK_SIZE_M=128, BLOCK_SIZE_N=512, BLOCK_SIZE_K=128,
        TOTAL_TILES=6, GROUP_SIZE_M=group_size_m, USE_ADDR_OFFSET_INT64_A=wide, USE_ADDR_OFFSET_INT64_B=wide,
        USE_ADDR_OFFSET_INT64_C=wide, INSN_WARMUP=False, num_tokens_post_padded_ptr=count)
    compiled = gluon.gemm_top_w8a8_channel_wasp_persistent[(1,)](
        a, b, sa, sb, actual, ids, experts, m, e, n, k, topk, *sb.stride(), *sa.stride(), **options)
    calls = []
    original = moe_op.fused_moe_kernel
    class TrackedTriton:
        def __getitem__(self, grid):
            launch = original[grid]
            def run(*args, **kwargs):
                calls.append(True)
                return launch(*args, **kwargs)
            return run
    monkeypatch.setattr(moe_op, "_try_gluon_channel_gemm1", lambda *a, **kw: False)
    monkeypatch.setattr(moe_op, "_USE_MOE_PERSISTENT_KERNEL", False)
    monkeypatch.setattr(moe_op, "splitk_size", 0)
    monkeypatch.setattr(moe_op, "fused_moe_kernel", TrackedTriton())
    moe_op.fused_moe(a, b, ordinary, sa, sb, None, None,
        torch.zeros((m, topk), dtype=torch.int32, device="cuda"), ids, None, experts, count,
        False, topk, tl.bfloat16 if output_kind == "bf16" else tl.float16,
        use_fp8_w8a8=kind.startswith("fp8"), use_int8_w8a8=kind == "int8", per_channel_quant=True,
        c_sorted=True, ck_sorting=True, ck_topk=topk,
        config=dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=group_size_m,
                    COMBINE_SCALE_LOAD=False, USE_MLS_LOAD=False, num_warps=4, num_stages=2))
    assert len(calls) == 1
    g, t = actual.cpu(), ordinary.cpu()
    torch.testing.assert_close(g, golden, rtol=0, atol=0)
    torch.testing.assert_close(t, golden, rtol=0, atol=0)
    torch.testing.assert_close(g, t, rtol=0, atol=0)
    report = dict(kind=kind, output=output_kind, forced_int64=wide, group_size_m=group_size_m, elements=g.numel(),
                  differing_elements=int((g != t).sum()), shared_bytes=compiled.metadata.shared)
    record_property("equivalence", json.dumps(report))
    print(json.dumps(report), flush=True)


@pytest.mark.parametrize("cu_count", [1, 48, 96, 192])
@pytest.mark.parametrize("kind", ["fp8", "int8"])
def test_bn512_support_query(cu_count, kind):
    from boltops.fused_moe.gluon import gemm_top_w8a8_channel_wasp_persistent_bm128xbn512xbk128 as gluon
    dtype = torch.int8 if kind == "int8" else torch.float8_e5m2
    a = torch.empty((97, 1024), dtype=dtype, device="meta")
    b = torch.empty((2, 1536, 1024), dtype=dtype, device="meta")
    sa = torch.empty((97, 1), device="meta")
    sb = torch.empty((2, 1536, 1), device="meta")
    opts = dict(arch="gfx946", cu_count=cu_count, top_k=2,
                config=dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=512, BLOCK_SIZE_K=128),
                scale_recipe="int8_channel" if kind == "int8" else "fp8_channel")
    assert gluon.is_persistent_channel_gemm1_supported(a, b, sa, sb, **opts)
    for change in (dict(ck_sorting=False), dict(c_sorted=False), dict(a_sorted=True),
                   dict(has_bias=True), dict(mul_routed_weight=True), dict(cu_count=0)):
        assert not gluon.is_persistent_channel_gemm1_supported(a, b, sa, sb, **dict(opts, **change))


@pytest.mark.parametrize("n,k,expected_bn", [(512, 1024, 512), (1536, 512, 512),
                                             (512, 768, 256), (768, 1024, 256)])
def test_framework_selects_bn512_and_preserves_legacy_fallback(monkeypatch, n, k, expected_bn):
    from boltops.utility import arch_info
    from boltops.utility.arch_info import DeviceInfo
    from boltops.fused_moe.triton import moe_op
    from boltops.fused_moe.gluon import gemm_top_w8a8_channel_wasp_persistent_bm128xbn512xbk128 as wide
    from boltops.fused_moe.gluon import gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 as legacy
    import triton.language as tl
    monkeypatch.setattr(arch_info, "get_arch", lambda *a: "gfx946")
    monkeypatch.setattr(arch_info, "get_device_info", lambda *a: DeviceInfo(0, "gfx946", 96))
    launches = []
    class Capture:
        def __getitem__(self, grid):
            def run(*args, **opts): launches.append((grid, opts))
            return run
    monkeypatch.setattr(wide, "gemm_top_w8a8_channel_wasp_persistent", Capture())
    monkeypatch.setattr(legacy, "gemm_top_w8a8_channel_wasp_persistent", Capture())
    a = torch.empty((97, k), dtype=torch.float8_e4m3fn, device="meta")
    b = torch.empty((2, n, k), dtype=a.dtype, device="meta")
    c = torch.empty((384, n), dtype=torch.bfloat16, device="meta")
    sa = torch.empty((97, 1), device="meta")
    sb = torch.empty((2, n, 1), device="meta")
    ids = torch.empty((384,), dtype=torch.int32, device="meta")
    experts = torch.empty((3,), dtype=torch.int32, device="meta")
    count = torch.empty((1,), dtype=torch.int32, device="meta")
    moe_op.fused_moe(a, b, c, sa, sb, None, None, torch.empty((97, 2), dtype=torch.int32, device="meta"),
        ids, None, experts, count, False, 2, tl.bfloat16, use_fp8_w8a8=True,
        per_channel_quant=True, c_sorted=True, ck_sorting=True, ck_topk=2,
        config=dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
                    GROUP_SIZE_M=1, num_warps=4, num_stages=2))
    assert len(launches) == 1
    grid, opts = launches[0]
    assert grid == (96,)
    assert opts["BLOCK_SIZE_N"] == expected_bn
    assert opts["TOTAL_TILES"] == 3 * ((n + expected_bn - 1) // expected_bn)
    assert opts["num_tokens_post_padded_ptr"] is count
