import os,json,time,sys
from pathlib import Path
import torch
from boltops.tools.gluon_moe_gemm1 import prepare_cpu,save_kernel
from boltops.fused_moe.triton import moe_op
from boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm64xbn512xbk256 import gemm_top_w8a8_channel_wasp_persistent as kernel, DEFAULT_LAUNCH_OPTIONS
# Real repository kernel and host dispatcher; no module/launch-option adapter.
r=Path.cwd()
M=int(os.environ.get('MOE_M','1024'));BM=64;BN=int(os.environ['MOE_BN']);BK=int(os.environ['MOE_BK']);E=256;N=1024;K=2048;TOPK=8
case=dict(M=M,E=E,N=512,K=K,topk=TOPK)
fixture=Path('/zhenxin/temp/bm64-gemm1-port-20261116/frozen_inputs.pt')
if fixture.exists():
 t,meta=torch.load(fixture,weights_only=False)
else:
 t,_,_,_,meta=prepare_cpu(case,'fp8_channel',BM,BN,0,3101)
 torch.save((t,meta),fixture)
if os.environ.get('MOE_PREPARE_ONLY')=='1':sys.exit(0)
rows=t['Sorted'].numel();full_total=t['Experts'].numel()*((N+BN-1)//BN)
total=96;grid=48;assert full_total>=total
gpu={n:x.cuda() for n,x in t.items()}
output=torch.full((rows,N),float('nan'),dtype=torch.bfloat16,device='cpu').cuda()
sa,sb=t['SA'],t['SB']
args=(gpu['A'],gpu['B'],gpu['SA'],gpu['SB'],output,gpu['Sorted'],gpu['Experts'],M,E,N,K,TOPK,*sb.stride(),*sa.stride())
opts=dict(DEFAULT_LAUNCH_OPTIONS)
opts.update(A_STAGES=2,PARTITION_REGS=(32,48,216,216),PERSISTENT_GRID=grid,TOTAL_TILES=total,BLOCK_SIZE_M=BM,BLOCK_SIZE_N=BN,BLOCK_SIZE_K=BK,USE_ADDR_OFFSET_INT64_A=False,USE_ADDR_OFFSET_INT64_B=False,USE_ADDR_OFFSET_INT64_C=False,INSN_WARMUP=False)
launch=dict(case=case,output_n=N,grid=[grid],logical_output_tiles=total,warmup_block_N=128,warmup_blocks_per_consumer=2,full_logical_tiles=full_total,options=opts,input_sha256=meta['input_sha256'],route_counts=meta['local_route_counts'])
(r/'launch.json').write_text(json.dumps(launch,indent=2));print('CPU data prepared',launch,flush=True)
(r/'gemm').mkdir(exist_ok=True);os.chdir(r/'gemm');padded_gpu=torch.tensor([total//((N+BN-1)//BN)*BM],dtype=torch.int32,device='cpu').cuda();opts['num_tokens_post_padded_ptr']=padded_gpu;compiled=kernel.warmup(*args,grid=(grid,),**opts);save_kernel(compiled)
if os.environ.get('MOE_COMPILE_ONLY')=='1':print('COMPILE ONLY',flush=True);sys.exit(0)
os.chdir(r)
padded=torch.tensor([total//((N+BN-1)//BN)*BM],dtype=torch.int32,device='cpu').cuda()
kernel[(grid,)](*args, **opts)
torch.cuda.synchronize()
actual=output.cpu();written=torch.zeros_like(actual,dtype=torch.bool);errors=0;padded=0;checked=0;nan=0
nblocks=(N+BN-1)//BN
for tile in range(total):
 bm,bn=divmod(tile,nblocks);left=bn*BN;right=min(left+BN,N)
 tokens=(t['Sorted'][bm*BM:(bm+1)*BM].to(torch.int64)&0xffffff)
 expert=int(t['Experts'][bm]);valid=(tokens<M)&(expert>=0)&(expert<E)
 expected=((t['A'].float()[tokens[valid]]@t['B'][max(expert,0),left:right].float().T)*sa[tokens[valid]]*sb[max(expert,0),left:right,0][None,:]).bfloat16()
 got=actual[bm*BM:(bm+1)*BM,left:right]
 errors+=int((~torch.isclose(got[valid].float(),expected.float(),rtol=.02,atol=.002)).sum());checked+=expected.numel();padded+=int((got[~valid]!=0).sum());nan+=int(torch.isnan(got).sum());written[bm*BM:(bm+1)*BM,left:right]=True
res=dict(actual_dispatcher_used=False,frozen_algorithm='BN512-B3-single-ticket', experimental_kernel_snapshot=False,INSN_WARMUP=False,capacity_logical_tiles=full_total,selected_logical_tiles=total,equivalent_BN256_tiles=192,experimental_launch_option_adapter=False,GPU_padded_counter_used=True,mismatches=errors,padded_row_errors=padded,remaining_nan=nan,checked_valid_elements=checked,untouched_modified=int((~torch.isnan(actual[~written])).sum()),shared=compiled.metadata.shared)
(r/'result.json').write_text(json.dumps(res,indent=2));print(res,flush=True);assert errors==padded==nan==res['untouched_modified']==0
