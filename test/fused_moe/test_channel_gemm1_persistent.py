import re
import pytest
import torch

@pytest.mark.parametrize('bn,n,k,bk,a_stages,total_tiles,grid,warmup', [
    pytest.param(128,256,1024,128,2,3,1,True, id='bn128-mls'),
    (256,512,1024,128,2,3,1,True),
    (256,512,1024,128,2,4,1,True),
    (256,512,1024,128,2,5,2,True),
    (256,512,1024,128,2,1,2,True),
    (128,256,1024,128,2,5,2,False),
    (128,248,1024,128,2,6,1,True),
    pytest.param(256,248,1024,128,2,3,2,False,id="bn256-mls-tail"),
    (256,512,1024,128,2,5,1,True),
])
@pytest.mark.parametrize('output_dtype', [torch.bfloat16, torch.float16], ids=['bf16', 'fp16'])
def test_persistent_sorted_tiles(bn,n,k,bk,a_stages,total_tiles,grid,warmup,output_dtype,topk=2,a_dtype=torch.float8_e4m3fn,b_dtype=None,b_scale_2d=False,broadcast_scales=None,entry_launcher=None,extra_output_blocks=0,int8_inputs=False,route_blocks=3):
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import gemm_top_w8a8_channel_wasp_persistent as kernel, DEFAULT_LAUNCH_OPTIONS
    rng=torch.Generator(device='cpu').manual_seed(3197)
    m,e,bm=97,2,128
    a=(torch.randint(-8,9,(m,k),generator=rng).float()/8).to(a_dtype)
    b=(torch.randint(-8,9,(e,n,k),generator=rng).float()/8).to(a.dtype if b_dtype is None else b_dtype)
    if int8_inputs:
        a=torch.randint(-127,128,(m,k),generator=rng,dtype=torch.int8)
        b=torch.randint(-127,128,(e,n,k),generator=rng,dtype=torch.int8)
    sa=torch.full((m,3),float('nan'));sb=torch.full((e,n,3),float('nan'))
    sa[:,1]=torch.randint(1,5,(m,),generator=rng).float()/8
    sb[:,:,1]=torch.randint(1,5,(e,n),generator=rng).float()/8
    blocks = []
    for _ in range(route_blocks):
        ids = torch.randperm(m*topk,generator=rng,dtype=torch.int32)[:bm]
        blocks.append(torch.cat((ids, torch.full((bm-ids.numel(),), m*topk, dtype=torch.int32))))
    sorted_ids = torch.cat(blocks)
    sorted_ids[3:8]=torch.tensor([-1,m*topk,m*topk+1,-2,2**31-1],dtype=torch.int32)
    if topk > 128:
        sorted_ids[0] = topk - 1  # Force a valid negative packed int32 ID.
    # Keep flattened IDs for the independent CPU reference.
    valid_routes = (sorted_ids >= 0) & (sorted_ids < m * topk)
    packed_ids = ((sorted_ids // topk) | ((sorted_ids % topk) << 24)).to(torch.int32)
    packed_ids = torch.where(valid_routes, packed_ids, m)
    if topk > 128:
        assert packed_ids[0] < 0
    experts=torch.tensor(([1,0,-1]*((route_blocks+2)//3))[:route_blocks],dtype=torch.int32)
    if broadcast_scales in ("a","all"):
        sa=sa[:1].expand(m,-1)
    if broadcast_scales in ("expert","all"):
        sb=sb[:1].expand(e,-1,-1)
    if broadcast_scales in ("channel","all"):
        sb=sb[:,:1].expand(-1,n,-1)
    sa_gpu=sa.cuda()[:,1:2];sb_gpu=sb.cuda()[:,:,1:2]
    if broadcast_scales in ("a","all"):
        sa_gpu=sa_gpu[:1].expand(m,-1)
    if broadcast_scales in ("expert","all"):
        sb_gpu=sb_gpu[:1].expand(e,-1,-1)
    if broadcast_scales in ("channel","all"):
        sb_gpu=sb_gpu[:,:1].expand(-1,n,-1)
    if b_scale_2d:
        sb_gpu=sb_gpu.squeeze(-1)
    b_scale_strides=(*sb_gpu.stride(),0) if sb_gpu.ndim==2 else sb_gpu.stride()
    output=torch.full(((route_blocks+extra_output_blocks)*bm,n),float('nan'),dtype=output_dtype).cuda()
    args=(a.cuda(),b.cuda(),sa_gpu,sb_gpu,output,packed_ids.cuda(),experts.cuda(),m,e,n,k,topk,*b_scale_strides,*sa_gpu.stride())
    opts=dict(USE_ADDR_OFFSET_INT64_A=a.numel() * a.element_size() >= 2**32 - 1,USE_ADDR_OFFSET_INT64_B=b.numel() * b.element_size() >= 2**32 - 1,USE_ADDR_OFFSET_INT64_C=output.numel() * output.element_size() >= 2**32 - 1,TOTAL_TILES=total_tiles,BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,A_STAGES=a_stages,INSN_WARMUP=warmup,PARTITION_REGS=None,num_warps=4,num_stages=2,use_async_copy=True,wasp_partition_warps=(4,4,4,4),wasp_wdra=True)
    opts.update(DEFAULT_LAUNCH_OPTIONS)
    if opts["USE_ADDR_OFFSET_INT64_A"]:
        opts["amdgpu_enable_max_ilp_scheduling_strategy"] = False
    compiled=kernel.warmup(*args,grid=(grid,),**opts)
    asm=compiled.asm['amdgcn']
    assert compiled.metadata.shared == 128 * 1024
    assert int(re.search(r'\.sgpr_spill_count:\s*(\d+)', asm)[1]) == 0
    assert int(re.search(r'\.vgpr_spill_count:\s*(\d+)', asm)[1]) == 0
    assert int(re.search(r'; ScratchSize:\s*(\d+)', asm)[1]) == 0
    if int8_inputs:
        assert re.search(r'v_mmac_i32_.*i8',asm)
        assert not re.search(r'v_mmac_f32_.*fp8',asm)
    # Every geometry uses the eight-slot ticket protocol and four SIMD handoffs.
    counts=re.findall(r'^\s*s_ebarrier_(?:sync|arrive)\s+\S+,\s*(\d+)',asm,re.M)
    assert '2' in counts
    assert not re.search(r'^\s*s_ebarrier_(?:sync|arrive)\s+\S+,\s*12(?:\s|$)',asm,re.M)
    assert re.search(r'^\s*s_ebarrier_(?:sync|arrive)\s+5,\s*16',asm,re.M)
    labels=re.findall(r'^\s*\.Lmoe_iwarm_c[01]_b[01]_mmac:',asm,re.M)
    assert len(labels)==(4 if warmup else 0)
    if entry_launcher is None:
        kernel[(grid,)](*args,**opts)
    else:
        entry_launcher(args,opts)
    actual=output.cpu();written=torch.zeros_like(actual,dtype=torch.bool)
    for tile in range(total_tiles):
        row,col=divmod(tile,(n+bn-1)//bn);left=col*bn;right=min(left+bn,n)
        routes=sorted_ids[row*bm:(row+1)*bm].long();expert=int(experts[row])
        valid=(routes>=0)&(routes<m*topk)&(expert>=0)&(expert<e)
        expected=((a.float()[routes[valid]//topk]@b[max(expert,0),left:right].float().T)*sa[routes[valid]//topk,1:2]*sb[max(expert,0),left:right,1][None,:]).to(output_dtype)
        got=actual[row*bm:(row+1)*bm,left:right]
        torch.testing.assert_close(got[valid],expected,rtol=.02,atol=.002)
        assert (got[~valid]==0).all()
        written[row*bm:(row+1)*bm,left:right]=True
    assert torch.isnan(actual[~written]).all()


@pytest.mark.parametrize("bn", [128, 256])
@pytest.mark.parametrize("int8_inputs", [False, True], ids=["fp8", "int8"])
def test_persistent_offsets_above_4g(bn, int8_inputs):
    """Exercise real >4-GiB relative offsets with small physical allocations.

    Virtual bases are shifted backwards; route/expert/tile indices compensate
    exactly. Every executed memory access still targets an allocated tensor.
    """
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import (
        gemm_top_w8a8_channel_wasp_persistent as kernel, DEFAULT_LAUNCH_OPTIONS)

    class RebasedPointer:
        def __init__(self, tensor, byte_shift):
            self.tensor = tensor  # Keep the allocation alive.
            self.dtype = tensor.dtype
            self.device = tensor.device
            self.address = tensor.data_ptr() - byte_shift

        def data_ptr(self):
            return self.address

    bm, bk, k = 128, 128, 1024
    rng = torch.Generator(device="cpu").manual_seed(3411)
    a = torch.randint(-8, 9, (bm, k), generator=rng)
    b = torch.randint(-8, 9, (1, bn, k), generator=rng)
    if int8_inputs:
        a, b = a.to(torch.int8), b.to(torch.int8)
    else:
        a = (a.float() / 8).to(torch.float8_e4m3fn)
        b = (b.float() / 8).to(a.dtype)
    sa = torch.full((bm, 1), .125)
    sb = torch.full((1, bn, 1), .25)
    a_bias = 2**32 // k + bm
    b_bias = 2**32 // (bn * k) + 1
    c_bias = 2**32 // (bm * bn * 2) + 1
    routes = (torch.arange(bm, dtype=torch.int32) + a_bias).cuda()
    experts = torch.tensor([b_bias], dtype=torch.int32).cuda()
    output = torch.full((bm, bn), float("nan"), dtype=torch.bfloat16).cuda()
    args = (
        RebasedPointer(a.cuda(), a_bias * k),
        RebasedPointer(b.cuda(), b_bias * bn * k),
        RebasedPointer(sa.cuda(), a_bias * 4),
        RebasedPointer(sb.cuda(), b_bias * bn * 4),
        RebasedPointer(output, c_bias * bm * bn * 2),
        RebasedPointer(routes, c_bias * bm * 4),
        RebasedPointer(experts, c_bias * 4),
        a_bias + bm, b_bias + 1, bn, k, 1, bn, 1, 1, 1, 1)
    options = dict(DEFAULT_LAUNCH_OPTIONS)
    options.update(TOTAL_TILES=c_bias + 1, TILE_ID=c_bias,
                   BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                   USE_ADDR_OFFSET_INT64_A=True, USE_ADDR_OFFSET_INT64_B=True,
                   USE_ADDR_OFFSET_INT64_C=True,
                   amdgpu_enable_max_ilp_scheduling_strategy=False)
    compiled = kernel.warmup(*args, grid=(1,), **options)
    # Use raw virtual bases with the compiled launcher's pointer interface.
    import inspect
    signature = inspect.signature(kernel.fn)
    bound = signature.bind(*args, **{key: value for key, value in options.items()
                                   if key in signature.parameters})
    bound.apply_defaults()
    raw_args = [value.data_ptr() if isinstance(value, RebasedPointer) else value
                for value in bound.arguments.values()]
    compiled[(1, 1, 1)](*raw_args)
    torch.cuda.synchronize()
    expected = ((a.float() @ b[0].float().T) * sa * sb[0, :, 0]).bfloat16()
    torch.testing.assert_close(output.cpu(), expected, rtol=.02, atol=.002)


@pytest.mark.parametrize("topk", [1, 129, 256])
def test_ck_sorting_token_bits(topk):
    # High-byte route indices >=128 make valid packed IDs negative as int32.
    test_persistent_sorted_tiles(128,248,1024,128,2,5,2,False,
                                 torch.bfloat16,topk=topk)


@pytest.mark.parametrize("a_dtype,b_dtype", [
    (torch.float8_e5m2, torch.float8_e5m2),
    (torch.float8_e4m3fn, torch.float8_e5m2),
    (torch.float8_e5m2, torch.float8_e4m3fn),
])
@pytest.mark.parametrize('bn,n', [(128,256),(256,512),(128,248),(256,248)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16,torch.float16])
def test_fp8_operand_formats(a_dtype,b_dtype,bn,n,output_dtype):
    test_persistent_sorted_tiles(bn,n,1024,128,2,3,1,False,
                                 output_dtype,a_dtype=a_dtype,b_dtype=b_dtype)


@pytest.mark.parametrize("output_dtype", [torch.bfloat16,torch.float16])
def test_channel_scale_2d_weight(output_dtype):
    test_persistent_sorted_tiles(128,256,1024,128,2,3,1,False,
                                 output_dtype,b_scale_2d=True)


@pytest.mark.parametrize("broadcast_scales", ["a","expert","channel","all"])
@pytest.mark.parametrize("bn", [128,256])
def test_channel_broadcast_scales(broadcast_scales,bn):
    test_persistent_sorted_tiles(bn,512,1024,128,2,3,1,False,
                                 torch.bfloat16,broadcast_scales=broadcast_scales)


@pytest.mark.parametrize("cu_budget", [1,48])
@pytest.mark.parametrize("n,group_size_m", [(128,1),(248,1),(256,1),(512,2)])
def test_moe_entry_gluon_device_work_count(monkeypatch,cu_budget,n,group_size_m,int8_inputs=False):
    import dataclasses
    import triton.language as tl
    from boltops.utility import arch_info
    from boltops.fused_moe.triton import moe_op

    info=arch_info.get_device_info()
    monkeypatch.setattr(arch_info,"get_device_info",lambda *args: dataclasses.replace(info,cu_count=cu_budget))
    class UnexpectedFallback:
        def __getitem__(self,grid):
            raise AssertionError("eligible GEMM1 unexpectedly fell back to ordinary Triton")
    monkeypatch.setattr(moe_op,"fused_moe_kernel",UnexpectedFallback())
    def launch(args,opts):
        a,b,sa,sb,c,ids,experts=args[:7]
        m,_,n,_,topk=args[7:12]
        # Capacity includes two valid-looking extra blocks. The GPU count must
        # prevent any read/store for them, leaving the extra output rows NaN.
        ids=torch.cat((ids.cpu(),torch.zeros(256,dtype=torch.int32))).cuda()
        experts=torch.cat((experts.cpu(),torch.zeros(2,dtype=torch.int32))).cuda()
        count=torch.tensor([384],dtype=torch.int32).cuda()
        topids=torch.empty((m,topk),dtype=torch.int32).cuda()
        moe_op.fused_moe(
            a,b,c,sa,sb,None,None,topids,ids,None,experts,count,False,topk,tl.bfloat16,
            use_fp8_w8a8=not int8_inputs,use_int8_w8a8=int8_inputs,per_channel_quant=True,c_sorted=True,ck_sorting=True,ck_topk=topk,
            config=dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=64,BLOCK_SIZE_K=64,
                        GROUP_SIZE_M=group_size_m,num_warps=4,num_stages=2))
    bn=256 if n>=256 and n%128==0 else 128
    test_persistent_sorted_tiles(bn,n,1024,128,2,3*((n+bn-1)//bn),1,False,
                                 torch.bfloat16,entry_launcher=launch,extra_output_blocks=2,int8_inputs=int8_inputs)


def test_moe_entry_triton_fallback_numerical(monkeypatch):
    import triton.language as tl
    from boltops.fused_moe.triton import moe_op
    from boltops.fused_moe.gluon import gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 as gluon_module

    class UnexpectedGluon:
        def __getitem__(self,grid):
            raise AssertionError("K640 must fall back to ordinary Triton")
    monkeypatch.setattr(gluon_module,"gemm_top_w8a8_channel_wasp_persistent",UnexpectedGluon())
    monkeypatch.setattr(moe_op,"_USE_MOE_PERSISTENT_KERNEL",False)
    monkeypatch.setattr(moe_op,"splitk_size",0)
    m,n,k=128,128,640
    rng=torch.Generator(device="cpu").manual_seed(3479)
    a=(torch.randint(-8,9,(m,k),generator=rng).float()/8).to(torch.float8_e4m3fn)
    b=(torch.randint(-8,9,(1,n,k),generator=rng).float()/8).to(a.dtype)
    sa=torch.full((m,1),.125);sb=torch.full((1,n),.25)
    c=torch.full((m,n),float("nan"),dtype=torch.bfloat16).cuda()
    ids=torch.arange(m,dtype=torch.int32).cuda()
    experts=torch.zeros(1,dtype=torch.int32).cuda()
    count=torch.tensor([m],dtype=torch.int32).cuda()
    topids=torch.zeros((m,1),dtype=torch.int32).cuda()
    moe_op.fused_moe(a.cuda(),b.cuda(),c,sa.cuda(),sb.cuda(),None,None,topids,
                    ids,None,experts,count,False,1,tl.bfloat16,
                    use_fp8_w8a8=True,per_channel_quant=True,c_sorted=True,ck_sorting=True,ck_topk=1,
                    config=dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=64,BLOCK_SIZE_K=64,
                                GROUP_SIZE_M=1,num_warps=4,num_stages=2))
    expected=((a.float()@b[0].float().T)*sa*sb[0]).bfloat16()
    torch.testing.assert_close(c.cpu(),expected,rtol=.02,atol=.002)


@pytest.mark.parametrize("shared_ring", [False, True])
def test_channel_scale_sparse_stride_above_4g(shared_ring):
    """A/B data remain small/32-bit; A's scale alone spans over 4 GiB."""
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import (
        gemm_top_w8a8_channel_wasp_persistent as kernel, DEFAULT_LAUNCH_OPTIONS)

    class RebasedScale:
        def __init__(self, tensor, shift):
            self.tensor, self.dtype, self.device = tensor, tensor.dtype, tensor.device
            self.address = tensor.data_ptr() - shift
        def data_ptr(self):
            return self.address

    m,n,k,bm=2,(256 if shared_ring else 128),(1024 if shared_ring else 512),128
    stride=2**30+1  # FP32 byte offset exceeds 4 GiB at token 1.
    a=torch.ones((m,k),dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
    b=torch.ones((1,n,k),dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
    sa=torch.tensor([.125],dtype=torch.float32).cuda()
    sb=torch.full((1,n),.25).cuda()
    c=torch.full((bm,n),float("nan"),dtype=torch.bfloat16).cuda()
    routes=torch.full((bm,),1,dtype=torch.int32).cuda()
    experts=torch.zeros(1,dtype=torch.int32).cuda()
    scale_a,scale_b=RebasedScale(sa,stride*4),sb
    stride_a,stride_b,tile=stride,1,0
    args=(a,b,scale_a,scale_b,c,routes,experts,m,1,n,k,1,n*stride_b,stride_b,0,stride_a,0)
    opts=dict(DEFAULT_LAUNCH_OPTIONS,TOTAL_TILES=tile+1,TILE_ID=tile,BLOCK_SIZE_M=bm,BLOCK_SIZE_N=256 if shared_ring else 128,
              BLOCK_SIZE_K=128,USE_ADDR_OFFSET_INT64_A=False,
              USE_ADDR_OFFSET_INT64_B=False,USE_ADDR_OFFSET_INT64_C=False,INSN_WARMUP=False)
    compiled=kernel.warmup(*args,grid=(1,),**opts)
    import inspect
    signature=inspect.signature(kernel.fn)
    bound=signature.bind(*args,**{key:value for key,value in opts.items() if key in signature.parameters})
    bound.apply_defaults()
    compiled[(1,1,1)](*[v.data_ptr() if isinstance(v,RebasedScale) else v for v in bound.arguments.values()])
    torch.cuda.synchronize()
    actual=c.cpu()
    torch.testing.assert_close(actual[:,tile*128:],torch.full((bm,n-tile*128),k/32,dtype=torch.bfloat16))
    assert torch.isnan(actual[:,:tile*128]).all()


@pytest.mark.parametrize('bn,n', [(128,256),(256,512),(128,248),(256,248)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16,torch.float16])
def test_int8_channel_tiles(bn,n,output_dtype):
    test_persistent_sorted_tiles(bn,n,1024,128,2,3*((n+bn-1)//bn),1,True,output_dtype,int8_inputs=True)


@pytest.mark.parametrize("n", [128,248,256])
@pytest.mark.parametrize("cu_budget", [1,48])
def test_int8_channel_entry(monkeypatch,n,cu_budget):
    test_moe_entry_gluon_device_work_count(monkeypatch,cu_budget,n,2,True)


@pytest.mark.parametrize("k", [512,768,1024,1280,2048,3072,4096])
def test_shared_ring_long_persistence(k):
    # Sixteen outputs/TG exercise every even BK remainder modulo8, rotating
    # metadata/scale slots and both abarrier phases across outputs.
    test_persistent_sorted_tiles(256,512,k,128,2,16,1,True,torch.bfloat16,route_blocks=8)


@pytest.mark.parametrize("n,k", [(384,1024),(512,512),(512,768)])
def test_shared_ring_partial_n_and_short_k(n,k):
    # Half-filled N blocks and incomplete eight-slot K cycles remain supported.
    test_persistent_sorted_tiles(256,n,k,128,2,6,1,False,torch.bfloat16)


def test_shared_ring_zero_device_work():
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import (
        gemm_top_w8a8_channel_wasp_persistent as kernel, DEFAULT_LAUNCH_OPTIONS)
    a=torch.ones((1,1024)).to(torch.float8_e4m3fn).cuda()
    b=torch.ones((1,256,1024)).to(a.dtype).cuda()
    sa=torch.ones((1,1)).cuda();sb=torch.ones((1,256)).cuda()
    c=torch.full((128,256),float("nan"),dtype=torch.bfloat16).cuda()
    ids=torch.zeros(128,dtype=torch.int32).cuda()
    experts=torch.zeros(1,dtype=torch.int32).cuda()
    count=torch.zeros(1,dtype=torch.int32).cuda()
    opts=dict(DEFAULT_LAUNCH_OPTIONS,TOTAL_TILES=1,BLOCK_SIZE_M=128,
              BLOCK_SIZE_N=256,BLOCK_SIZE_K=128,USE_ADDR_OFFSET_INT64_A=False,
              USE_ADDR_OFFSET_INT64_B=False,USE_ADDR_OFFSET_INT64_C=False,
              num_tokens_post_padded_ptr=count)
    kernel[(48,)](a,b,sa,sb,c,ids,experts,1,1,256,1024,1,256,1,0,1,0,**opts)
    assert torch.isnan(c.cpu()).all()


@pytest.mark.parametrize("topk", [1,129,256])
def test_shared_ring_ck_route_bits(topk):
    test_persistent_sorted_tiles(256,512,1024,128,2,6,1,False,
                                 torch.bfloat16,topk=topk)


def test_shared_ring_wide_output_base():
    from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128 import (
        gemm_top_w8a8_channel_wasp_persistent as kernel, DEFAULT_LAUNCH_OPTIONS)
    class Shifted:
        def __init__(self,tensor,shift):
            self.tensor,self.dtype,self.device=tensor,tensor.dtype,tensor.device
            self.address=tensor.data_ptr()-shift
        def data_ptr(self):return self.address
    bm,bn,k=128,256,1024
    block=2**32//(bm*bn*2)+1
    a=torch.ones((bm,k)).to(torch.float8_e4m3fn).cuda()
    b=torch.ones((1,bn,k)).to(a.dtype).cuda()
    sa=torch.full((bm,1),.125).cuda();sb=torch.full((1,bn),.25).cuda()
    c=torch.full((bm,bn),float("nan"),dtype=torch.bfloat16).cuda()
    routes=torch.arange(bm,dtype=torch.int32).cuda()
    experts=torch.zeros(1,dtype=torch.int32).cuda()
    args=(a,b,sa,sb,Shifted(c,block*bm*bn*2),Shifted(routes,block*bm*4),
          Shifted(experts,block*4),bm,1,bn,k,1,bn,1,0,1,0)
    opts=dict(DEFAULT_LAUNCH_OPTIONS,TOTAL_TILES=block+1,TILE_ID=block,
              BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=128,
              USE_ADDR_OFFSET_INT64_A=False,USE_ADDR_OFFSET_INT64_B=False,
              USE_ADDR_OFFSET_INT64_C=True)
    compiled=kernel.warmup(*args,grid=(1,),**opts)
    import inspect
    signature=inspect.signature(kernel.fn)
    bound=signature.bind(*args,**{key:value for key,value in opts.items() if key in signature.parameters})
    bound.apply_defaults()
    compiled[(1,1,1)](*[v.data_ptr() if isinstance(v,Shifted) else v for v in bound.arguments.values()])
    torch.cuda.synchronize()
    torch.testing.assert_close(c.cpu(),torch.full((bm,bn),32.,dtype=torch.bfloat16))


@pytest.mark.parametrize("bn,n", [(128,248),(256,248)])
@pytest.mark.parametrize("int8_inputs", [False,True], ids=["fp8","int8"])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16,torch.float16], ids=["bf16","fp16"])
def test_mls_boundary_tail(bn,n,int8_inputs,output_dtype):
    # Multiple persistent outputs reuse tail slots and exercise absent halves.
    test_persistent_sorted_tiles(bn,n,1024,128,2,4*((n+bn-1)//bn),1,
                                 False,output_dtype,int8_inputs=int8_inputs,route_blocks=4)
