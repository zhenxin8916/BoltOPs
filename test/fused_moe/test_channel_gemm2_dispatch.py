# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Metadata-only coverage for the optional BN256 channel GEMM2 candidate."""
import importlib
import pytest
import torch

MODULE = "boltops.fused_moe.gluon.gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn256xbk128"

@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8])
@pytest.mark.parametrize("cu", [1, 48, 96, 192])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float16])
def test_supports_channel_types_and_cu_budget(dtype, cu, output):
    mod = importlib.import_module(MODULE)
    a = torch.empty((256, 256), dtype=dtype, device="meta")
    b = torch.empty((2, 512, 256), dtype=dtype, device="meta")
    sa = torch.empty((256, 1), device="meta")
    sb = torch.empty((2, 512, 1), device="meta")
    assert mod.is_persistent_channel_gemm2_supported(
        a, b, sa, sb, arch="gfx946", cu_count=cu, top_k=2,
        output_dtype=output, config=dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64),
        scale_recipe="int8_channel" if dtype == torch.int8 else "fp8_channel")

@pytest.mark.parametrize("mul", [True, False])
def test_dispatch_uses_ck_topk_and_preserves_config(monkeypatch, mul):
    from boltops.fused_moe.triton import moe_op
    from boltops.utility.arch_info import DeviceInfo
    mod = importlib.import_module(MODULE)
    monkeypatch.setattr(moe_op.arch_info, "get_device_info", lambda *_: DeviceInfo(0, "gfx946", 96))
    launches = []
    class Capture:
        def __getitem__(self, grid):
            def run(*args, **kwargs): launches.append((grid, args, kwargs))
            return run
    monkeypatch.setattr(mod, "gemm_bottom_w8a8_channel_wasp_persistent", Capture())
    def tensor(shape, dtype=torch.float32): return torch.empty(shape, dtype=dtype, device="meta")
    a=tensor((256,256),torch.int8);b=tensor((2,512,256),torch.int8)
    c=tensor((97,2,512),torch.float16);sa=tensor((256,1));sb=tensor((2,512,1))
    ids=tensor((256,),torch.int32);exp=tensor((2,),torch.int32);count=tensor((1,),torch.int32)
    weights=tensor((256,)) if mul else None
    config=dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64)
    assert moe_op._try_gluon_channel_gemm2(a,b,c,sa,sb,weights,ids,exp,count,1,config,
        use_fp8_w8a8=False,use_int8_w8a8=True,per_channel_quant=True,block_shape=None,
        c_sorted=False,a_sorted=True,ck_sorting=True,ck_topk=2,naive_block_assignment=False,
        mul_routed_weight=mul,B_bias=None,B_zp=None)
    grid,args,opts=launches[0]
    assert grid==(96,) and args[5] is weights and args[-1]==4
    assert opts["top_k"]==2 and opts["MUL_ROUTED_WEIGHT"]==mul
    assert opts["num_tokens_post_padded_ptr"] is count
    assert "N_TILES_LIMIT" not in opts
    assert config==dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=64,BLOCK_SIZE_K=64)

@pytest.mark.parametrize("changes", [dict(a_sorted=False),dict(c_sorted=True),dict(ck_sorting=False),dict(has_bias=True),dict(arch="gfx938")])
def test_unsupported_cases_fall_back(changes):
    mod=importlib.import_module(MODULE)
    a=torch.empty((256,256),dtype=torch.int8,device="meta");b=torch.empty((2,512,256),dtype=torch.int8,device="meta")
    sa=torch.empty((256,1),device="meta");sb=torch.empty((2,512,1),device="meta")
    opts=dict(arch="gfx946",cu_count=48,top_k=2,config=dict(BLOCK_SIZE_M=128),scale_recipe="int8_channel")
    opts.update(changes)
    assert not mod.is_persistent_channel_gemm2_supported(a,b,sa,sb,**opts)

@pytest.mark.parametrize("arch,expected", [("gfx946", True), ("gfx938", False)])
def test_sorted_bridge_selection_precedes_activation(monkeypatch, arch, expected):
    from boltops.fused_moe.triton import moe_op
    from boltops.utility.arch_info import DeviceInfo
    monkeypatch.setattr(moe_op.arch_info, "get_device_info", lambda *_: DeviceInfo(0, arch, 48))
    b = torch.empty((2, 512, 256), dtype=torch.int8, device="meta")
    sb = torch.empty((2, 512, 1), device="meta")
    assert moe_op._gluon_gemm2_prefers_sorted_layout(
        b, sb, top_k=2, config=dict(BLOCK_SIZE_M=128), output_dtype=torch.bfloat16,
        use_fp8_w8a8=False, use_int8_w8a8=True, per_channel_quant=True,
        block_shape=None, ck_sorting=True, has_bias=False) == expected
