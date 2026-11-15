# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Compare Gluon to ordinary Triton using immutable AIDSL input/golden files."""

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("bn", [128, 256])
@pytest.mark.parametrize("kind", ["fp8", "int8"])
@pytest.mark.parametrize("output_kind", ["bf16", "fp16"])
def test_frozen_gluon_matches_triton(
    monkeypatch, bn, kind, output_kind, record_property
):
    import triton.language as tl
    from boltops.fused_moe.gluon import (
        gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 as gluon,
    )
    from boltops.fused_moe.triton import moe_op
    from boltops.tools.export_moe_gemm1 import load_frozen_case

    value = os.environ.get("AIDSL_MOE_CASE_ROOT")
    if not value:
        pytest.skip("Set AIDSL_MOE_CASE_ROOT to the exported frozen AIDSL cases")
    directory = Path(value) / f"{kind}_bn{bn}_{output_kind}_m97_n256_k512_grid1"
    cfg, host = load_frozen_case(directory)
    golden = host.pop("C")
    gpu = {name: tensor.cuda() for name, tensor in host.items()}
    a, b, sa, sb, ids, experts, count = (
        gpu[name] for name in ("A", "B", "SA", "SB", "Sorted", "Experts", "WorkCount")
    )
    params = cfg["parameters"]
    m, n, k, topk = (params[name] for name in ("m", "n", "k", "topk"))
    total_tiles = experts.numel() * ((n + bn - 1) // bn)
    # Ordinary Triton requires pre-zeroed padding/invalid-expert output rows.
    # Use that public contract for both backends, then compare every output.
    actual_gluon = torch.zeros_like(golden, device="cuda")
    actual_triton = torch.zeros_like(actual_gluon)
    options = dict(
        gluon.DEFAULT_LAUNCH_OPTIONS,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=128,
        TOTAL_TILES=total_tiles,
        USE_ADDR_OFFSET_INT64_A=False,
        USE_ADDR_OFFSET_INT64_B=False,
        USE_ADDR_OFFSET_INT64_C=False,
        num_tokens_post_padded_ptr=count,
    )
    gluon.gemm_top_w8a8_channel_wasp_persistent[(1,)](
        a,
        b,
        sa,
        sb,
        actual_gluon,
        ids,
        experts,
        m,
        params["experts"],
        n,
        k,
        topk,
        *sb.stride(),
        0,
        *sa.stride(),
        **options,
    )

    calls = []
    original_kernel = moe_op.fused_moe_kernel

    class TrackedTriton:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def run(*args, **kwargs):
                calls.append(True)
                return launch(*args, **kwargs)

            return run

    # Explicitly force the ordinary kernel; two Gluon dispatches are not a comparison.
    monkeypatch.setattr(
        moe_op, "_try_gluon_channel_gemm1", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(moe_op, "_USE_MOE_PERSISTENT_KERNEL", False)
    monkeypatch.setattr(moe_op, "splitk_size", 0)
    monkeypatch.setattr(moe_op, "fused_moe_kernel", TrackedTriton())
    # This API uses topk_ids only for its shape/valid-token count. Actual routing
    # for both kernels comes from the same frozen CK Sorted/Experts buffers.
    topids = torch.zeros((m, topk), device="cuda", dtype=torch.int32)
    moe_op.fused_moe(
        a,
        b,
        actual_triton,
        sa,
        sb,
        None,
        None,
        topids,
        ids,
        None,
        experts,
        count,
        False,
        topk,
        tl.bfloat16 if output_kind == "bf16" else tl.float16,
        use_fp8_w8a8=kind == "fp8",
        use_int8_w8a8=kind == "int8",
        per_channel_quant=True,
        c_sorted=True,
        ck_sorting=True,
        ck_topk=topk,
        config=dict(
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=1,
            COMBINE_SCALE_LOAD=False,
            USE_MLS_LOAD=False,
            num_warps=4,
            num_stages=2,
        ),
    )
    assert len(calls) == 1
    g, t = actual_gluon.cpu(), actual_triton.cpu()
    assert torch.isfinite(g).all() and torch.isfinite(t).all()
    torch.testing.assert_close(g, golden, rtol=0.02, atol=0.002)
    torch.testing.assert_close(t, golden, rtol=0.02, atol=0.002)
    torch.testing.assert_close(g, t, rtol=0, atol=0)
    report = dict(
        case=directory.name,
        elements=g.numel(),
        differing_elements=int((g != t).sum()),
        max_abs_difference=float((g.float() - t.float()).abs().max()),
        input_sha256={
            name: desc["sha256"]
            for name, desc in cfg["buffers"].items()
            if desc["role"] == "input"
        },
    )
    record_property("equivalence", json.dumps(report))
    print(json.dumps(report), flush=True)
    # Loading again verifies no test/backend modified the fixed files.
    load_frozen_case(directory)
