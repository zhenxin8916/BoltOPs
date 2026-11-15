"""Retest unchanged kernels with frozen inputs and equal per-TG work."""
import argparse,hashlib,importlib,json,os
from pathlib import Path
import torch
from boltops.tools.gluon_moe_gemm1 import save_kernel
p=argparse.ArgumentParser();p.add_argument('stage',choices=('top','bottom'));p.add_argument('bn',type=int);p.add_argument('kind',choices=('fp8','int8'));p.add_argument('--warm',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parent
stage,bn,kind=args.stage,args.bn,args.kind
source_name=('gemm_top_w8a8_channel_wasp_persistent_bm128xbn512xbk128' if bn==512 else 'gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128') if stage=='top' else ('gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn512xbk64' if bn==512 else 'gemm_bottom_w8a8_channel_wasp_persistent_bm128xbn256xbk128')
mod=importlib.import_module('boltops.fused_moe.gluon.'+source_name)
data=Path('/zhenxin/temp/moe-gemm1-bn512-port-20261115') if stage=='top' else Path('/zhenxin/temp/moe-gemm2-bn512-port-20261115' if bn==512 else '/zhenxin/temp/moe-gemm2-bn256-csv-20261115')
frozen=torch.load(data/f'frozen-{kind}.pt',map_location='cpu',weights_only=False);manifest=json.loads((data/f'frozen-{kind}.json').read_text());host=frozen['host']
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
assert all(sha(t)==manifest['input_sha256'][n] for n,t in host.items())
gpu={n:t.cuda() for n,t in host.items()};per_tg=1024//bn;tiles=48*per_tg
if stage=='top':
 bk=128;output=torch.full((host['Sorted'].numel(),1024),float('nan'),dtype=torch.bfloat16).cuda()
 kernel=mod.gemm_top_w8a8_channel_wasp_persistent
 arguments=(gpu['A'],gpu['B'],gpu['SA'],gpu['SB'],output,gpu['Sorted'],gpu['Experts'],4096,256,1024,2048,8,*host['SB'].stride(),*host['SA'].stride())
 options=dict(mod.DEFAULT_LAUNCH_OPTIONS,TOTAL_TILES=tiles,BLOCK_SIZE_M=128,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,INSN_WARMUP=args.warm)
else:
 bk=64 if bn==512 else 128;output=torch.full((32768,2048),float('nan'),dtype=torch.bfloat16).cuda()
 kernel=mod.gemm_bottom_w8a8_channel_wasp_persistent
 arguments=(gpu['A'],gpu['B'],gpu['SA'],gpu['SB'],output,gpu['SortedWeights'],gpu['Sorted'],gpu['Experts'],host['Sorted'].numel(),32768,tiles*(2048//bn))
 options=dict(mod.DEFAULT_LAUNCH_OPTIONS,N=2048,K=256,top_k=8,BLOCK_SIZE_M=128,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,CK_SORTING=True,N_TILES_LIMIT=1)
options.update(USE_ADDR_OFFSET_INT64_A=False, USE_ADDR_OFFSET_INT64_B=False, USE_ADDR_OFFSET_INT64_C=False)
compiled=kernel.warmup(*arguments,grid=(48,),**options)
Path('gemm').mkdir(exist_ok=True);os.chdir('gemm');save_kernel(compiled);os.chdir('..')
launch=dict(stage=stage,bn=bn,bm=128,bk=bk,kind=kind,warmup=int(args.warm),grid=48,cu_count=48,tiles_per_tg=per_tg,total_tiles=tiles,output_elements_per_tg=128*1024,source_path=mod.__file__,source_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),input_sha256=manifest['input_sha256'],frozen_path=str(data),options=options,clang_path=os.environ['TRITON_HIP_CLANG_PATH'],gpu_buffers={n:dict(address=t.data_ptr(),bytes=t.numel()*t.element_size()) for n,t in dict(gpu,C=output).items()})
Path('launch.json').write_text(json.dumps(launch,indent=2)+'\n')
kernel[(48,)](*arguments,**options)
actual=output.cpu()
if stage=='top':
 got=actual[:6144];expected=frozen['golden'];untouched=bool(torch.isnan(actual[6144:]).all())
else:
 mask=frozen['selected_rows'];got=actual[mask,:bn];expected=frozen['golden'][mask];untouched=bool(torch.isnan(actual[:,bn:]).all() and torch.isnan(actual[~mask,:bn]).all())
errors=int((~torch.isclose(got.float(),expected.float(),rtol=.02,atol=.002)).sum())
result=dict(verified=errors==0 and untouched,mismatches=errors,checked_elements=got.numel(),untouched_tail=untouched,output_sha256=sha(got),shared=compiled.metadata.shared)
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
assert result['verified'],result
