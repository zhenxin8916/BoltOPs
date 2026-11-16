import importlib.util,sys,os,json,hashlib
from pathlib import Path
import torch
from boltops.tools.gluon_moe_gemm1 import prepare_cpu,save_kernel
source=Path(sys.argv[1]);kind=sys.argv[2] if len(sys.argv)>2 else 'fp8'
wide=os.environ.get('CHECK_WIDE')=='1'
fp16=os.environ.get('CHECK_FP16')=='1'
out_dtype=torch.float16 if fp16 else torch.bfloat16
spec=importlib.util.spec_from_file_location('check_candidate',source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
case=dict(M=193,E=5,N=256,K=512,topk=2)
t,*_=prepare_cpu(case,'fp8_channel',64,256,0,4107,gemm='gemm2')
encoded=t['Sorted'].long();routes=(encoded&0xffffff)*2+((encoded>>24)&255);valid=routes<386;gather=routes.clamp_max(385)
if kind == 'int8':
    t['A']=(t['A'].float()*16).round().clamp(-127,127).to(torch.int8)
    t['B']=(t['B'].float()*16).round().clamp(-127,127).to(torch.int8)
    t['SA']/=16;t['SB']/=16
sa=t['SA'][gather].clone();sa[~valid]=0;t['SA']=sa
x=t['A'].view(torch.uint8)[gather].clone();x[~valid]=0;t['A']=x.view(t['A'].dtype)
rows=routes.numel();gold=torch.full((386,512),float('nan'),dtype=out_dtype)
for m in range(rows//64):
    span=slice(m*64,(m+1)*64);mask=valid[span];rt=routes[span][mask];e=int(t['Experts'][m])
    if e>=0:
        value=(((t['A'][span].float()[mask]@t['B'][e].float().T)*t['SA'][span][mask])*t['SB'][e,:,0][None,:])*t['RouteWeights'][rt,None]
        gold[rt]=value.to(out_dtype)
gpu={n:x.cuda() for n,x in t.items()};c=torch.full((386,512),float('nan'),dtype=out_dtype).cuda()
opts=dict(mod.DEFAULT_LAUNCH_OPTIONS,N=512,K=256,top_k=2,BLOCK_SIZE_M=64,BLOCK_SIZE_N=256,BLOCK_SIZE_K=int(os.environ.get("BENCH_BK", "128")),CK_SORTING=True,USE_ADDR_OFFSET_INT64_A=wide,USE_ADDR_OFFSET_INT64_B=wide,USE_ADDR_OFFSET_INT64_C=wide)
args=(gpu['A'],gpu['B'],gpu['SA'],gpu['SB'],c,gpu['SortedWeights'],gpu['Sorted'],gpu['Experts'],rows,386,rows//64*2)
k=mod.gemm_bottom_w8a8_channel_wasp_persistent.warmup(*args,grid=(2,),**opts)
Path('gemm').mkdir(exist_ok=True);os.chdir('gemm');save_kernel(k);os.chdir('..')
mod.gemm_bottom_w8a8_channel_wasp_persistent[(2,)](*args,**opts)

if os.environ.get('COMPARE_TRITON') == '1':
    from boltops.fused_moe.triton.moe_op import fused_moe_persistent_kernel
    import triton.language as tl
    ref = torch.full_like(c, float('nan'))
    padded = torch.tensor([rows], dtype=torch.int32).cuda()
    aa,bb,ssa,ssb = gpu['A'],gpu['B'],gpu['SA'],gpu['SB']
    arguments = (aa,bb,ref,gpu['RouteWeights'],padded,gpu['Experts'],gpu['Sorted'],
                 gpu['SortedWeights'],ssa,ssb,512,256,rows,386,
                 *aa.stride(),bb.stride(0),bb.stride(2),bb.stride(1),512,1,
                 *ssa.stride(),ssb.stride(0),ssb.stride(2),ssb.stride(1),rows)
    opts_ref = dict(BLOCK_SIZE_M=64,BLOCK_SIZE_N=256,BLOCK_SIZE_K=128,GROUP_SIZE_M=1,
        COMBINE_SCALE_LOAD=False,USE_MLS_LOAD=False,MUL_ROUTED_WEIGHT=True,
        USE_ADDR_OFFSET_INT64_A=wide,USE_ADDR_OFFSET_INT64_B=wide,USE_ADDR_OFFSET_INT64_C=wide,
        top_k=1,compute_type=tl.float16 if fp16 else tl.bfloat16,
        use_fp8_w8a8=kind=='fp8',use_int8_w8a8=kind=='int8',use_int8_w8a16=False,
        per_channel_quant=True,c_sorted=False,a_sorted=True,bottom_a_use_mls_load=False,
        ck_sorting=True,ck_topk=2,NUM_SMS=2,NUM_XCDS=1,group_n=0,group_k=0,
        num_warps=4,num_stages=1,kpack=2,use_block_pingpong=False,buffer_cache_swizzle=True)
    fused_moe_persistent_kernel[(2,)](*arguments, **opts_ref)
    reference = ref.cpu()
    check_got = c.cpu()
    triton_errors=int((~torch.isclose(check_got.float(),reference.float(),rtol=.02,atol=.002)).sum())
    bitwise_unequal=int((check_got.view(torch.int16)!=reference.view(torch.int16)).sum())
    Path('triton-comparison.json').write_text(json.dumps(dict(mismatches=triton_errors,bitwise_unequal=bitwise_unequal,rtol=.02,atol=.002,
        checked_elements=check_got.numel(),kind=kind,wide=wide,fp16=fp16),indent=2))
    assert triton_errors == 0, triton_errors

got=c.cpu();errors=int((~torch.isclose(got.float(),gold.float(),rtol=.02,atol=.002)).sum())
result=dict(kind=kind,wide=wide,fp16=fp16,mismatches=errors,checked_elements=got.numel(),rows=rows,m_blocks=rows//64,grid=2,n_tiles_per_m=2,full_N=True,shared=k.metadata.shared)
Path('result.json').write_text(json.dumps(result,indent=2));print(result,flush=True);assert errors==0
