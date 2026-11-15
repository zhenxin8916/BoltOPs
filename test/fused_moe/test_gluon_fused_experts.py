"""Persistent channel GEMM1 layout selection."""
import pytest


@pytest.mark.parametrize("m,n,waves,mma_tiles_per_wave_n,expected", [
    (128, 64, 4, 1, (2, 2)),
    (64, 128, 4, 1, (1, 4)),
    (128, 64, 4, 2, (4, 1)),
    (16, 64, 4, 1, (1, 4)),
    (64, 16, 4, 1, (4, 1)),
    (16, 16, 1, 1, (1, 1)),
    (16, 64, 2, 1, (1, 2)),
])
def test_gemm1_mma_warps(m, n, waves, mma_tiles_per_wave_n, expected):
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import gemm1_mma_warps
    assert gemm1_mma_warps(m, n, waves, mma_tiles_per_wave_n) == expected


@pytest.mark.parametrize("n,expected", [(64, (4, 1)), (128, (2, 2))])
def test_gemm1_mma_warps_default(n, expected):
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import gemm1_mma_warps
    assert gemm1_mma_warps(128, n) == expected


@pytest.mark.parametrize("bn,n,expected", [
    (128, 256, True),
    (128, 248, True),
    (128, 256, True),
    (128, 248, True),
    (256, 256, True),
    (256, 248, True),
    (256, 248, True),
])
@pytest.mark.parametrize("output_dtype", [None, "bfloat16", "float16"])
def test_channel_dispatch_supports_bn128_and_bn256(bn, n, expected, output_dtype):
    import torch
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import is_persistent_channel_gemm1_supported

    # Meta tensors let the CPU query check shapes/dtypes/strides without GPU work.
    a = torch.empty((97, 1024), dtype=torch.float8_e4m3fn, device="meta")
    b = torch.empty((2, n, 1024), dtype=a.dtype, device="meta")
    sa = torch.empty((97, 3), device="meta")[:, 1:2]
    sb = torch.empty((2, n, 3), device="meta")[:, :, 1:2]
    config = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128)
    dtype = None if output_dtype is None else getattr(torch, output_dtype)
    assert is_persistent_channel_gemm1_supported(a, b, sa, sb, arch="gfx946", cu_count=48, top_k=2, config=config,
                        output_dtype=dtype) is expected


@pytest.mark.parametrize("change", [
    {"arch": "gfx938"}, {"cu_count": 0}, {"cu_count": -1}, {"top_k": 0},
    {"c_sorted": False}, {"a_sorted": True}, {"ck_sorting": False}, {"top_k": 257}, {"gemm_type": "gemm2"},
    {"scale_recipe": "fp8_block"}, {"has_bias": True}, {"mul_routed_weight": True},
    {"output_dtype": "float32"}, {"a_dtype": "float8_e4m3fnuz"},
    {"scale_dtype": "float16"}, {"k": 640}, {"a_column_major": True},
    {"bad_scale_shape": True}, {"config": {"BLOCK_SIZE_K": 64}},
    {"config": {"BLOCK_SIZE_M": 64}},
])
def test_channel_dispatch_rejects_unsupported_scenarios(change):
    import torch
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import is_persistent_channel_gemm1_supported

    change = change.copy()
    k = change.pop("k", 1024)
    a = torch.empty((97, k), dtype=getattr(torch, change.pop("a_dtype", "float8_e4m3fn")), device="meta")
    if change.pop("a_column_major", False):
        a = a.as_strided(a.shape, (1, 97))
    b = torch.empty((2, 256, k), dtype=torch.float8_e4m3fn, device="meta")
    scale_dtype = getattr(torch, change.pop("scale_dtype", "float32"))
    sa = torch.empty((97, 2 if change.pop("bad_scale_shape", False) else 1), dtype=scale_dtype, device="meta")
    sb = torch.empty((2, 256, 1), dtype=scale_dtype, device="meta")
    config = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128)
    config.update(change.pop("config", {}))
    opts = dict(arch="gfx946", cu_count=48, top_k=2, config=config)
    if "output_dtype" in change:
        change["output_dtype"] = getattr(torch, change["output_dtype"])
    opts.update(change)
    assert not is_persistent_channel_gemm1_supported(a, b, sa, sb, **opts)


@pytest.mark.parametrize("cu_count", [1, 48, 96, 192])
def test_channel_dispatch_accepts_caller_cu_budget(cu_count):
    import torch
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import is_persistent_channel_gemm1_supported

    a = torch.empty((97, 1024), dtype=torch.float8_e4m3fn, device="meta")
    b = torch.empty((2, 256, 1024), dtype=a.dtype, device="meta")
    sa = torch.empty((97, 1), device="meta")
    sb = torch.empty((2, 256, 1), device="meta")
    config = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128)
    assert is_persistent_channel_gemm1_supported(a, b, sa, sb, arch="gfx946", cu_count=cu_count, top_k=2, config=config)


@pytest.mark.parametrize("m,n,k", [
    (2**22 + 1, 256, 1024),  # A exceeds 4 GiB.
    (97, 2**22 + 256, 1024),  # One expert B exceeds 4 GiB, MLS.
])
def test_channel_dispatch_large_addresses(m, n, k):
    import torch
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import is_persistent_channel_gemm1_supported

    a = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="meta")
    b = torch.empty((2, n, k), dtype=a.dtype, device="meta")
    sa = torch.empty((m, 1), device="meta")
    sb = torch.empty((2, n, 1), device="meta")
    config = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128)
    assert is_persistent_channel_gemm1_supported(a, b, sa, sb, arch="gfx946", cu_count=48, top_k=2, config=config)


@pytest.mark.parametrize("internal_config", [
    {"A_STAGES": 3}, {"num_warps": 8}, {"num_stages": 4},
    {"COPY_BYTES": 8}, {"LDS_WRAP": 64},
    {"wasp_partition_warps": (4, 4, 8, 8)},
])
def test_channel_dispatch_does_not_validate_internal_launch_config(internal_config):
    import torch
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import is_persistent_channel_gemm1_supported

    a = torch.empty((97, 1024), dtype=torch.float8_e4m3fn, device="meta")
    b = torch.empty((2, 256, 1024), dtype=a.dtype, device="meta")
    sa = torch.empty((97, 1), device="meta")
    sb = torch.empty((2, 256, 1), device="meta")
    config = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128, **internal_config)
    assert is_persistent_channel_gemm1_supported(a, b, sa, sb, arch="gfx946", cu_count=48, top_k=2, config=config)


@pytest.mark.parametrize("a_dtype,b_dtype", [
    ("float8_e5m2","float8_e5m2"),
    ("float8_e4m3fn","float8_e5m2"),
    ("float8_e5m2","float8_e4m3fn"),
])
@pytest.mark.parametrize("bn", [128,256])
@pytest.mark.parametrize("output_dtype", ["bfloat16","float16"])
@pytest.mark.parametrize("b_scale_ndim", [2,3])
def test_channel_dispatch_fp8_formats(a_dtype,b_dtype,bn,output_dtype,b_scale_ndim):
    import torch
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import is_persistent_channel_gemm1_supported

    a=torch.empty((97,1024),dtype=getattr(torch,a_dtype),device="meta")
    b=torch.empty((2,256,1024),dtype=getattr(torch,b_dtype),device="meta")
    sa=torch.empty((97,1),device="meta")
    sb=torch.empty((2,256) if b_scale_ndim==2 else (2,256,1),device="meta")
    assert is_persistent_channel_gemm1_supported(
        a,b,sa,sb,arch="gfx946",cu_count=48,top_k=2,
        config=dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=128),
        output_dtype=getattr(torch,output_dtype))


@pytest.mark.parametrize("m", [2**24-1,2**24,2**24+1])
def test_channel_dispatch_does_not_add_ck_token_limit(m):
    import torch
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import is_persistent_channel_gemm1_supported

    a=torch.empty((m,1024),dtype=torch.float8_e4m3fn,device="meta")
    b=torch.empty((2,256,1024),dtype=a.dtype,device="meta")
    sa=torch.empty((m,1),device="meta");sb=torch.empty((2,256),device="meta")
    assert is_persistent_channel_gemm1_supported(
        a,b,sa,sb,arch="gfx946",cu_count=48,top_k=2,
        config=dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=256,BLOCK_SIZE_K=128))


@pytest.mark.parametrize("broadcast_scales", ["a","expert","channel","all"])
@pytest.mark.parametrize("bn", [128,256])
@pytest.mark.parametrize("b_scale_ndim", [2,3])
def test_channel_dispatch_broadcast_scales(broadcast_scales,bn,b_scale_ndim):
    import torch
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import is_persistent_channel_gemm1_supported

    a=torch.empty((97,1024),dtype=torch.float8_e4m3fn,device="meta")
    b=torch.empty((2,512,1024),dtype=a.dtype,device="meta")
    sa=torch.empty((1 if broadcast_scales in ("a","all") else 97,1),device="meta").expand(97,1)
    sb=torch.empty((1 if broadcast_scales in ("expert","all") else 2,
                    1 if broadcast_scales in ("channel","all") else 512),device="meta").expand(2,512)
    if b_scale_ndim==3:
        sb=sb.unsqueeze(-1)
    assert is_persistent_channel_gemm1_supported(
        a,b,sa,sb,arch="gfx946",cu_count=48,top_k=2,
        config=dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=128))


@pytest.mark.parametrize("change,expected", [
    ({},"gluon"), ({"n":128},"gluon"), ({"n":248},"gluon"), ({"group_size_m":4},"gluon"),
    ({"n_tiles":2,"legacy_persistent":True},"gluon"),
    ({"bottom_a_use_mls_load":True},"triton"),
    ({"int8":True},"gluon"), ({"int8":True,"n":128},"gluon"),
    ({"int8":True,"n":248},"gluon"),
    ({"arch":"gfx938"},"triton"), ({"no_gluon":True},"triton"),
    ({"k":640},"triton"), ({"bm":64},"triton"),
    ({"c_sorted":False},"triton"), ({"ck_sorting":False},"triton"),
    ({"bias":True},"triton"), ({"mul_routed_weight":True},"triton"),
])
def test_moe_entry_queries_gluon_before_triton(monkeypatch,change,expected):
    import importlib
    import torch
    import triton.language as tl
    from boltops.utility import arch_info
    from boltops.utility.arch_info import DeviceInfo

    monkeypatch.setattr(arch_info,"get_arch",lambda *args: "gfx946")
    monkeypatch.setattr(arch_info,"get_device_info",lambda *args: DeviceInfo(0,change.get("arch","gfx946"),96))
    moe=importlib.import_module("boltops.fused_moe.triton.moe_op")
    gluon=importlib.import_module("boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128")
    if change.get("no_gluon"):
        import importlib.util
        monkeypatch.setattr(importlib.util,"find_spec",lambda name: None)
    launches=[]
    class Capture:
        def __init__(self,name):self.name=name
        def __getitem__(self,grid):
            def launch(*args,**opts): launches.append((self.name,grid,opts))
            return launch
    monkeypatch.setattr(gluon,"gemm_top_w8a8_channel_wasp_persistent",Capture("gluon"))
    monkeypatch.setattr(moe,"fused_moe_kernel",Capture("triton"))
    monkeypatch.setattr(moe,"_USE_MOE_PERSISTENT_KERNEL",change.get("legacy_persistent",False))
    monkeypatch.setattr(moe,"splitk_size",0)
    k=change.get("k",1024);m,e,n,topk=97,2,change.get("n",256),2
    a=torch.empty((m,k),dtype=torch.int8 if change.get("int8") else torch.float8_e4m3fn,device="meta")
    b=torch.empty((e,n,k),dtype=a.dtype,device="meta")
    c=torch.empty((384,n),dtype=torch.bfloat16,device="meta")
    sa=torch.empty((m,1),device="meta");sb=torch.empty((e,n),device="meta")
    ids=torch.empty(384,dtype=torch.int32,device="meta")
    experts=torch.empty(3,dtype=torch.int32,device="meta")
    count=torch.empty(1,dtype=torch.int32,device="meta")
    weights=torch.empty((m,topk),device="meta")
    topids=torch.empty((m,topk),dtype=torch.int32,device="meta")
    config=dict(BLOCK_SIZE_M=change.get("bm",128),BLOCK_SIZE_N=64,BLOCK_SIZE_K=64,
                GROUP_SIZE_M=change.get("group_size_m",1),num_warps=4,num_stages=2)
    if "n_tiles" in change:
        config["N_TILES_PER_CTA"]=change["n_tiles"]
    original=config.copy()
    moe.fused_moe(a,b,c,sa,sb,None,weights,topids,ids,None,experts,count,
                 change.get("mul_routed_weight",False),topk,tl.bfloat16,
                 use_fp8_w8a8=not change.get("int8",False),use_int8_w8a8=change.get("int8",False),per_channel_quant=True,
                 c_sorted=change.get("c_sorted",True),ck_sorting=change.get("ck_sorting",True),
                 bottom_a_use_mls_load=change.get("bottom_a_use_mls_load",False),
                 ck_topk=topk,B_bias=torch.empty((e,n),device="meta") if change.get("bias") else None,
                 config=config)
    assert config==original
    assert len(launches)==1 and launches[0][0]==expected
    if expected=="gluon":
        _,grid,opts=launches[0]
        assert grid==(96,)
        bn=256 if n==256 else 128
        assert (opts["BLOCK_SIZE_M"],opts["BLOCK_SIZE_N"],opts["BLOCK_SIZE_K"])==(128,bn,128)
        assert opts["GROUP_SIZE_M"]==config["GROUP_SIZE_M"]
        assert opts["num_tokens_post_padded_ptr"] is count
        assert opts["TOTAL_TILES"]==3*((n+bn-1)//bn)
        assert "amdgpu_enable_max_ilp_scheduling_strategy" not in opts


@pytest.mark.parametrize("bn", [128,256])
@pytest.mark.parametrize("output_dtype", ["bfloat16","float16"])
@pytest.mark.parametrize("b_dtype,recipe,expected", [
    ("int8","int8_channel",True),
    ("float8_e4m3fn","int8_channel",False),
    ("int8","fp8_channel",False),
])
def test_int8_channel_support_query(bn,output_dtype,b_dtype,recipe,expected):
    import torch
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import is_persistent_channel_gemm1_supported
    a=torch.empty((97,1024),dtype=torch.int8,device="meta")
    b=torch.empty((2,512,1024),dtype=getattr(torch,b_dtype),device="meta")
    sa=torch.empty((97,1),device="meta");sb=torch.empty((2,512),device="meta")
    assert is_persistent_channel_gemm1_supported(
        a,b,sa,sb,arch="gfx946",cu_count=48,top_k=2,
        config=dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=128),
        scale_recipe=recipe,output_dtype=getattr(torch,output_dtype)) is expected
