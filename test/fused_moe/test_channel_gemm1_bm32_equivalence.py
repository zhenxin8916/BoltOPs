# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Opt-in BM32 GEMM1 equivalence on fixed CK routes and quantized inputs."""
import hashlib
import json
import os

import pytest
import torch


@pytest.mark.skipif(os.environ.get("RUN_GLUON_GEMM1_BM32_EQUIVALENCE") != "1",
                    reason="opt-in gfx946 numerical comparison")
@pytest.mark.parametrize("kind,out,wide", [
    ("e4m3", torch.bfloat16, False),
    ("int8", torch.bfloat16, False),
    ("e5m2", torch.bfloat16, False),
    ("mixed", torch.bfloat16, False),
    ("mixed_reverse", torch.float16, False),
    ("e4m3", torch.float16, True),
    ("int8", torch.float16, True),
    ("e4m3", torch.float16, "a"),
    ("e4m3", torch.float16, "b"),
    ("e4m3", torch.float16, "c"),
    ("grouped", torch.bfloat16, False),
    ("warmup", torch.bfloat16, False),
])
def test_gluon_gemm1_matches_triton(monkeypatch, record_property, kind, out, wide):
    import triton.language as tl
    from boltops.fused_moe.gluon import (
        gemm_top_w8a8_channel_wasp_persistent_bm32xbn256xbk256 as mod,
    )
    from boltops.fused_moe.triton.moe_op import fused_moe_kernel

    m, e, n, k, topk, rows = 128, 2, 256, 1024, 2, 64
    n = 512 if kind == "grouped" else n
    group_size = 4 if kind == "grouped" else 1
    grid = 2 if kind == "grouped" else 1
    flags = tuple(wide is True or wide == name for name in ("a", "b", "c"))
    gen = torch.Generator().manual_seed(3101)
    a = torch.randint(-4, 5, (m, k), generator=gen, dtype=torch.int8)
    b = torch.randint(-4, 5, (e, n, k), generator=gen, dtype=torch.int8)
    if kind != "int8":
        a = a.to(torch.float8_e5m2 if kind in ("e5m2", "mixed_reverse") else torch.float8_e4m3fn)
        b = b.to(torch.float8_e5m2 if kind in ("e5m2", "mixed") else torch.float8_e4m3fn)
    # Frozen routing: rank0/expert0 followed by rank1/expert1; include CK padding.
    tokens = torch.arange(rows, dtype=torch.int32) % 32
    tokens[28:32] = m
    ids = tokens | ((torch.arange(rows, dtype=torch.int32) // 32) << 24)
    experts = torch.tensor([0, 1], dtype=torch.int32)
    golden = torch.zeros((rows, n), dtype=out)
    for block in range(2):
        span = slice(block * 32, (block + 1) * 32)
        valid = tokens[span] < m
        value = (a.float()[tokens[span][valid].long()] @ b[block].float().T) / 64
        part = golden[span]
        part[valid] = value.to(out)
    sa = torch.full((m, 2 if wide else 1), .125)
    sb = torch.full((e, n, 2 if wide else 1), .125)
    A, B, SA, SB, S, E = [x.cuda() for x in (a, b, sa, sb, ids, experts)]
    if wide:
        SA, SB = SA[:, ::2], SB[:, :, ::2]
    count = torch.tensor([rows], dtype=torch.int32).cuda()
    weights = torch.ones((m * topk,), dtype=torch.float32).cuda()
    C = torch.zeros((rows, n), dtype=out).cuda()
    R = torch.zeros((rows, n), dtype=out).cuda()
    opts = dict(mod.DEFAULT_LAUNCH_OPTIONS, M=m, E=e, N=n, K=k, top_k=topk,
                stride_bse=SB.stride(0), stride_bsn=SB.stride(1), stride_bsk=SB.stride(2),
                stride_asm=SA.stride(0), stride_ask=SA.stride(1), TOTAL_TILES=5,
                BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256,
                USE_ADDR_OFFSET_INT64_A=flags[0], USE_ADDR_OFFSET_INT64_B=flags[1],
                USE_ADDR_OFFSET_INT64_C=flags[2], num_tokens_post_padded_ptr=count,
                PERSISTENT_GRID=grid, GROUP_SIZE_M=group_size, INSN_WARMUP=kind == "warmup")
    compiled = mod.gemm_top_w8a8_channel_wasp_persistent[(grid,)](A, B, SA, SB, C, S, E, **opts)
    asm = compiled.asm["amdgcn"]
    assert ".sgpr_spill_count: 0" in asm or ".sgpr_spill_count:     0" in asm
    assert ".vgpr_spill_count: 0" in asm or ".vgpr_spill_count:     0" in asm
    assert compiled.metadata.shared == 66 * 1024
    args = (A, B, R, weights, count, E, S, weights, SA, SB, B, n, k, rows, m * topk,
            *A.stride(), B.stride(0), B.stride(2), B.stride(1), *R.stride(),
            *SA.stride(), SB.stride(0), SB.stride(2), SB.stride(1), 0, 0, m)
    # Direct ordinary Triton invocation avoids the Gluon dispatcher entirely.
    fused_moe_kernel[((rows // 32) * (n // 128),)](
        *args, group_n=0, group_k=0, block_k_diviable=True, block_n_diviable=True,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=group_size,
        COMBINE_SCALE_LOAD=False, USE_MLS_LOAD=False, MUL_ROUTED_WEIGHT=False,
        USE_ADDR_OFFSET_INT64_A=flags[0], USE_ADDR_OFFSET_INT64_B=flags[1],
        USE_ADDR_OFFSET_INT64_C=flags[2], top_k=topk,
        compute_type=tl.float16 if out == torch.float16 else tl.bfloat16,
        use_fp8_w8a8=kind != "int8", use_int8_w8a8=kind == "int8", use_int8_w8a16=False,
        per_channel_quant=True, c_sorted=True, a_sorted=False, bottom_a_use_mls_load=False,
        ck_sorting=True, ck_topk=topk, naive_block_assignment=False, NUM_XCDS=1,
        SCALE_BIAS_WITH_ROUTED_WEIGHT=False, ADD_BIAS=False, single_k_tile=False,
        num_warps=4, num_stages=2)
    actual, ordinary = C.cpu(), R.cpu()
    torch.testing.assert_close(actual, golden, atol=0, rtol=0)
    torch.testing.assert_close(ordinary, golden, atol=0, rtol=0)
    torch.testing.assert_close(actual, ordinary, atol=0, rtol=0)
    if kind == "e4m3" and not wide:
        from types import SimpleNamespace
        from boltops.fused_moe.triton import moe_op
        framed = torch.zeros((rows, n), dtype=out).cuda()
        # Mock only the discovered CU count. The dispatcher and GPU kernel run.
        with monkeypatch.context() as ctx:
            ctx.setattr(moe_op.arch_info, "get_device_info",
                        lambda device: SimpleNamespace(arch="gfx946", cu_count=1))
            assert moe_op._try_gluon_channel_gemm1(
                A, B, framed, SA, SB, S, E, count, topk,
                dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256, GROUP_SIZE_M=1), use_fp8_w8a8=True,
                use_int8_w8a8=False, per_channel_quant=True, block_shape=None,
                c_sorted=True, a_sorted=False, ck_sorting=True, ck_topk=topk,
                naive_block_assignment=False, mul_routed_weight=False,
                B_bias=None, B_zp=None)
        torch.testing.assert_close(framed.cpu(), golden, atol=0, rtol=0)
    report = dict(kind=kind, output=str(out), forced_int64=flags, group_size_m=group_size, grid=grid,
                  elements=C.numel(), mismatches=int((actual != ordinary).sum()),
                  routing_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())
    record_property("equivalence", json.dumps(report))
    print(json.dumps(report), flush=True)
