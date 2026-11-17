# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Check BM32 ownership, address flags and ordinary-Triton fallback on metadata."""
import importlib

import pytest
import torch

def tensor(shape, dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device="meta")


class Capture:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.launches.append((grid, args, kwargs))
        return run


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8])
@pytest.mark.parametrize("cu_count", [48, 96])
def test_bm32_launch_uses_sorting_block_and_caller_cus(monkeypatch, stage, dtype, cu_count):
    from boltops.fused_moe.triton import moe_op
    from boltops.utility.arch_info import DeviceInfo
    monkeypatch.setattr(moe_op.arch_info, "get_device_info",
                        lambda *_: DeviceInfo(0, "gfx946", cu_count))
    bn, bk = (256, 256) if stage == 1 else (512, 64)
    stem = "top" if stage == 1 else "bottom"
    mod = importlib.import_module(
        f"boltops.fused_moe.gluon.gemm_{stem}_w8a8_channel_wasp_persistent_bm32xbn{bn}xbk{bk}")
    capture = Capture()
    monkeypatch.setattr(mod, f"gemm_{stem}_w8a8_channel_wasp_persistent", capture)
    k = 1024 if stage == 1 else 256
    a = tensor((128, k), dtype)
    b = tensor((2, 1024, k), dtype)
    c = tensor((128 if stage == 1 else 256, 1024), torch.bfloat16)
    sa, sb = tensor((128, 1)), tensor((2, 1024, 1))
    ids, owners, count = tensor((128,), torch.int32), tensor((4,), torch.int32), tensor((1,), torch.int32)
    cfg = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64)
    options = dict(use_fp8_w8a8=dtype != torch.int8, use_int8_w8a8=dtype == torch.int8,
                   per_channel_quant=True, block_shape=None, c_sorted=stage == 1,
                   a_sorted=stage == 2, ck_sorting=True, ck_topk=2,
                   naive_block_assignment=False, mul_routed_weight=False,
                   B_bias=None, B_zp=None)
    if stage == 1:
        accepted = moe_op._try_gluon_channel_gemm1(a,b,c,sa,sb,ids,owners,count,2,cfg,**options)
    else:
        assert moe_op._gluon_gemm2_prefers_sorted_layout(
            b,sb,top_k=2,config=cfg,output_dtype=c.dtype,
            use_fp8_w8a8=dtype != torch.int8,use_int8_w8a8=dtype == torch.int8,
            per_channel_quant=True,block_shape=None,ck_sorting=True,has_bias=False)
        accepted = moe_op._try_gluon_channel_gemm2(a,b,c,sa,sb,None,ids,owners,count,1,cfg,**options)
    assert accepted
    grid,args,opts = capture.launches[0]
    assert grid == (cu_count,)
    assert opts["num_tokens_post_padded_ptr"] is count
    if stage == 1:
        assert (opts["BLOCK_SIZE_M"],opts["BLOCK_SIZE_N"],opts["BLOCK_SIZE_K"]) == (32,bn,bk)
        assert opts["PERSISTENT_GRID"] == cu_count
        assert opts["TOTAL_TILES"] == 4*(1024//bn)
    else:
        assert args[-1] == 4*(1024//bn)
    assert cfg == dict(BLOCK_SIZE_M=32,BLOCK_SIZE_N=64,BLOCK_SIZE_K=64)


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("wide", ["A", "B", "C", "scale_A", "scale_B"])
def test_bm32_wide_address_selection(monkeypatch, stage, wide):
    from boltops.fused_moe.triton import moe_op
    from boltops.utility.arch_info import DeviceInfo
    monkeypatch.setattr(moe_op.arch_info, "get_device_info",lambda *_: DeviceInfo(0,"gfx946",48))
    bn,bk = (256,256) if stage == 1 else (512,64)
    stem = "top" if stage == 1 else "bottom"
    mod = importlib.import_module(
        f"boltops.fused_moe.gluon.gemm_{stem}_w8a8_channel_wasp_persistent_bm32xbn{bn}xbk{bk}")
    capture=Capture();monkeypatch.setattr(mod,f"gemm_{stem}_w8a8_channel_wasp_persistent",capture)
    # Keep GEMM1 M128 while using a large contiguous K to select its A64 path.
    k=(32*1024**2 if wide=="A" else 1024) if stage==1 else 256
    rows=32*1024**2 if wide=="A" and stage==2 else 128
    experts=65536 if wide=="B" else 2
    output_rows=16*1024**2 if wide=="C" else 256
    a,b=tensor((rows,k),torch.int8),tensor((experts,1024,k),torch.int8)
    c=tensor((output_rows,1024),torch.bfloat16)
    sa=(torch.empty_strided((rows,1),(1024**3,1),device="meta") if wide=="scale_A" else tensor((rows,1)))
    sb=(torch.empty_strided((experts,1024,1),(1024**4,1024**3,1),device="meta") if wide=="scale_B" else tensor((experts,1024,1)))
    ids,owners,count=tensor((128,),torch.int32),tensor((4,),torch.int32),tensor((1,),torch.int32)
    opts=dict(use_fp8_w8a8=False,use_int8_w8a8=True,per_channel_quant=True,
              block_shape=None,c_sorted=stage==1,a_sorted=stage==2,ck_sorting=True,
              ck_topk=2,naive_block_assignment=False,mul_routed_weight=False,B_bias=None,B_zp=None)
    if stage==1:
        ok=moe_op._try_gluon_channel_gemm1(a,b,c,sa,sb,ids,owners,count,2,dict(BLOCK_SIZE_M=32),**opts)
    else:
        ok=moe_op._try_gluon_channel_gemm2(a,b,c,sa,sb,None,ids,owners,count,1,dict(BLOCK_SIZE_M=32),**opts)
    assert ok
    launch=capture.launches[0][2]
    expected=wide[-1] if wide.startswith("scale_") else wide
    assert launch["USE_ADDR_OFFSET_INT64_"+expected]


@pytest.mark.parametrize("stage", [1, 2])
def test_bm32_unsupported_metadata_falls_back(stage):
    bn,bk = (256,256) if stage==1 else (512,64)
    stem="top" if stage==1 else "bottom"
    mod=importlib.import_module(
        f"boltops.fused_moe.gluon.gemm_{stem}_w8a8_channel_wasp_persistent_bm32xbn{bn}xbk{bk}")
    query=getattr(mod,f"is_persistent_channel_gemm{stage}_supported")
    k=1024 if stage==1 else 256
    args=(tensor((128,k),torch.int8),tensor((2,1024,k),torch.int8),tensor((128,1)),tensor((2,1024,1)))
    opts=dict(arch="gfx946",cu_count=48,top_k=2,
              config=dict(BLOCK_SIZE_M=32,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk),scale_recipe="int8_channel")
    assert query(*args,**opts)
    for change in (dict(arch="gfx942"),dict(ck_sorting=False),dict(has_bias=True),
                   dict(config=dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk))):
        assert not query(*args,**(opts|change))
