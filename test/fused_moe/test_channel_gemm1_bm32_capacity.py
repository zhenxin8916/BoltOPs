# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""BM32 GEMM1 work-count bounds with inactive TGs and valid backing capacity."""
import hashlib
import json
import os

import pytest
import torch


@pytest.mark.skipif(os.environ.get("RUN_GLUON_GEMM1_BM32_EQUIVALENCE") != "1",
                    reason="opt-in gfx946 numerical comparison")
@pytest.mark.parametrize("active_rows", [64, 0])
def test_gluon_gemm1_capacity_and_inactive_tgs(record_property, active_rows):
    import triton.language as tl
    from boltops.fused_moe.gluon import (
        gemm_top_w8a8_channel_wasp_persistent_bm32xbn256xbk256 as mod,
    )
    from boltops.fused_moe.triton.moe_op import fused_moe_kernel

    m, e, n, k, topk, bm = 128, 2, 256, 1024, 2, 32
    capacity_blocks, grid = 6, 8
    capacity_rows = capacity_blocks * bm
    generator = torch.Generator().manual_seed(3101)
    a = torch.randint(-4, 5, (m, k), generator=generator, dtype=torch.int8).to(torch.float8_e4m3fn)
    b = torch.randint(-4, 5, (e, n, k), generator=generator, dtype=torch.int8).to(torch.float8_e4m3fn)
    tokens = torch.full((capacity_rows,), m, dtype=torch.int32)
    tokens[:32] = torch.arange(32, dtype=torch.int32)
    tokens[32:64] = 48 + torch.arange(32, dtype=torch.int32)
    tokens[28:32] = m
    tokens[60:64] = m
    ranks = torch.zeros(capacity_rows, dtype=torch.int32)
    ranks[32:64] = 1
    ids = tokens | (ranks << 24)
    experts = torch.full((capacity_blocks,), -1, dtype=torch.int32)
    experts[:2] = torch.tensor([0, 1], dtype=torch.int32)
    sa = torch.full((m, 1), .125)
    sb = torch.full((e, n, 1), .125)
    # Full backing storage satisfies TOTAL_TILES even though most TGs are idle.
    # Distinct sentinels detect writes beyond the device-side active prefix.
    golden = torch.full((capacity_rows, n), 7, dtype=torch.bfloat16)
    for block in range(active_rows // bm):
        span = slice(block * bm, (block + 1) * bm)
        valid = tokens[span] < m
        part = golden[span]
        part.zero_()
        value = (a.float()[tokens[span][valid].long()] @ b[block].float().T) / 64
        part[valid] = value.bfloat16()
    A, B, SA, SB, S, E = [x.cuda() for x in (a, b, sa, sb, ids, experts)]
    count = torch.tensor([active_rows], dtype=torch.int32).cuda()
    C = torch.full((capacity_rows, n), 7, dtype=torch.bfloat16).cuda()
    R = torch.full((capacity_rows, n), 7, dtype=torch.bfloat16).cuda()
    weights = torch.ones((m * topk,), dtype=torch.float32).cuda()
    compiled = mod.gemm_top_w8a8_channel_wasp_persistent[(grid,)](
        A, B, SA, SB, C, S, E,
        M=m, E=e, N=n, K=k, top_k=topk,
        stride_bse=SB.stride(0), stride_bsn=SB.stride(1), stride_bsk=SB.stride(2),
        stride_asm=SA.stride(0), stride_ask=SA.stride(1), TOTAL_TILES=capacity_blocks,
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=n, BLOCK_SIZE_K=256,
        USE_ADDR_OFFSET_INT64_A=False, USE_ADDR_OFFSET_INT64_B=False,
        USE_ADDR_OFFSET_INT64_C=False, num_tokens_post_padded_ptr=count,
        PERSISTENT_GRID=grid, **mod.DEFAULT_LAUNCH_OPTIONS)
    args = (A, B, R, weights, count, E, S, weights, SA, SB, B, n, k,
            capacity_rows, m * topk, *A.stride(), B.stride(0), B.stride(2), B.stride(1),
            *R.stride(), *SA.stride(), SB.stride(0), SB.stride(2), SB.stride(1), 0, 0, m)
    # The ordinary path gets the same capacity and GPU counter, without dispatch mocks.
    fused_moe_kernel[(capacity_blocks * (n // 128),)](
        *args, group_n=0, group_k=0, block_k_diviable=True, block_n_diviable=True,
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1,
        COMBINE_SCALE_LOAD=False, USE_MLS_LOAD=False, MUL_ROUTED_WEIGHT=False,
        USE_ADDR_OFFSET_INT64_A=False, USE_ADDR_OFFSET_INT64_B=False,
        USE_ADDR_OFFSET_INT64_C=False, top_k=topk, compute_type=tl.bfloat16,
        use_fp8_w8a8=True, use_int8_w8a8=False, use_int8_w8a16=False,
        per_channel_quant=True, c_sorted=True, a_sorted=False,
        bottom_a_use_mls_load=False, ck_sorting=True, ck_topk=topk,
        naive_block_assignment=False, NUM_XCDS=1, SCALE_BIAS_WITH_ROUTED_WEIGHT=False,
        ADD_BIAS=False, single_k_tile=False, num_warps=4, num_stages=2)
    actual, ordinary = C.cpu(), R.cpu()
    torch.testing.assert_close(actual, golden, atol=0, rtol=0)
    active = torch.arange(capacity_rows) < active_rows
    valid = active & (tokens < m)
    # Ordinary Triton masks padded token stores, while Gluon writes padded zero.
    # Check both contracts without masking writes into inactive capacity rows.
    ordinary_golden = golden.clone()
    ordinary_golden[active & (tokens >= m)] = 7
    torch.testing.assert_close(ordinary, ordinary_golden, atol=0, rtol=0)
    torch.testing.assert_close(actual[valid], ordinary[valid], atol=0, rtol=0)
    assert torch.all(actual[active_rows:] == 7)
    assert torch.all(ordinary[active_rows:] == 7)
    report = dict(active_rows=active_rows, capacity_rows=capacity_rows,
                  total_tiles=capacity_blocks, grid=grid,
                  inactive_tgs=grid-active_rows//bm, elements=C.numel(),
                  mismatches=int((actual[valid] != ordinary[valid]).sum()),
                  padded_rows_zeroed_by_gluon=int((active & (tokens >= m)).sum()),
                  ordinary_padded_rows_unchanged=True,
                  inactive_elements_unchanged=(capacity_rows-active_rows)*n,
                  shared_bytes=compiled.metadata.shared,
                  routing_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())
    record_property("capacity_equivalence", json.dumps(report))
    print(json.dumps(report), flush=True)
