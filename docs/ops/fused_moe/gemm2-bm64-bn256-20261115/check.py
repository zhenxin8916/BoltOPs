import importlib.util,sys,os,json,hashlib
from pathlib import Path
import torch
from boltops.tools.gluon_moe_gemm1 import prepare_cpu,save_kernel
source=Path(sys.argv[1]);kind=sys.argv[2] if len(sys.argv)>2 else 'fp8'
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
rows=routes.numel();gold=torch.full((386,512),float('nan'),dtype=torch.bfloat16)
for m in range(rows//64):
    span=slice(m*64,(m+1)*64);mask=valid[span];rt=routes[span][mask];e=int(t['Experts'][m])
    if e>=0:
        value=(((t['A'][span].float()[mask]@t['B'][e].float().T)*t['SA'][span][mask])*t['SB'][e,:,0][None,:])*t['RouteWeights'][rt,None]
        gold[rt]=value.bfloat16()
gpu={n:x.cuda() for n,x in t.items()};c=torch.full((386,512),float('nan'),dtype=torch.bfloat16).cuda()
opts=dict(mod.DEFAULT_LAUNCH_OPTIONS,N=512,K=256,top_k=2,BLOCK_SIZE_M=64,BLOCK_SIZE_N=256,BLOCK_SIZE_K=int(os.environ.get("BENCH_BK", "128")),CK_SORTING=True)
args=(gpu['A'],gpu['B'],gpu['SA'],gpu['SB'],c,gpu['SortedWeights'],gpu['Sorted'],gpu['Experts'],rows,386,rows//64*2)
k=mod.gemm_bottom_w8a8_channel_wasp_persistent.warmup(*args,grid=(2,),**opts)
Path('gemm').mkdir(exist_ok=True);os.chdir('gemm');save_kernel(k);os.chdir('..')
mod.gemm_bottom_w8a8_channel_wasp_persistent[(2,)](*args,**opts)
got=c.cpu();errors=int((~torch.isclose(got.float(),gold.float(),rtol=.02,atol=.002)).sum())
result=dict(kind=kind,mismatches=errors,checked_elements=got.numel(),rows=rows,m_blocks=rows//64,grid=2,n_tiles_per_m=2,full_N=True,shared=k.metadata.shared)
Path('result.json').write_text(json.dumps(result,indent=2));print(result,flush=True);assert errors==0
