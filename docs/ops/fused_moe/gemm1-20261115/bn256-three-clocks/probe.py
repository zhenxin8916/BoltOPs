from pathlib import Path
import os,sys,json,hashlib,subprocess
import torch
from boltops.tools.export_moe_gemm1 import load_frozen_case
import importlib.util
import boltops.fused_moe.gluon
source_path=Path(os.environ['GEMM1_SOURCE'])
name='boltops.fused_moe.gluon.gemm_top_w8a8_channel_wasp_persistent_bm128xbn128_256xbk128'
spec=importlib.util.spec_from_file_location(name,source_path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
root=Path.cwd();kind,bn=sys.argv[1],int(sys.argv[2]);tiles_per_tg=1024//bn
case=Path('/zhenxin/temp/moe-chip-comparison-20261114/frozen-m4096')/kind
cfg,host=load_frozen_case(case);golden=host.pop('C');gpu={n:t.cuda() for n,t in host.items()};p=cfg['parameters']
M,E,N,K,topk=(p[n] for n in ('m','experts','n','k','topk'))
a,b,sa,sb,ids,experts,count=(gpu[n] for n in ('A','B','SA','SB','Sorted','Experts','WorkCount'))
c=torch.full_like(golden,float('nan'),device='cpu').cuda();props=torch.cuda.get_device_properties(0);grid=props.multi_processor_count
assert grid==48 and bn in (128,256)
tiles=grid*tiles_per_tg;output_rows=tiles//((N+bn-1)//bn)*128
assert output_rows==6144
opts=dict(mod.DEFAULT_LAUNCH_OPTIONS,BLOCK_SIZE_M=128,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=128,TOTAL_TILES=tiles,USE_ADDR_OFFSET_INT64_A=False,USE_ADDR_OFFSET_INT64_B=False,USE_ADDR_OFFSET_INT64_C=False,num_tokens_post_padded_ptr=count)
source=Path(mod.__file__);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
commit=subprocess.check_output(['git','-C','/zhenxin/BoltOPs','rev-parse','HEAD'],text=True).strip()
launch=dict(case=dict(M=M,E=E,N=N//2,K=K,topk=topk),actual_N=N,actual_K=K,grid=grid,physical_cus=grid,measured_logical_tiles=tiles,persistent_outputs_per_tg=tiles_per_tg,measured_output_rows=output_rows,diagnostic_only=True,input_sha256={k:v['sha256'] for k,v in cfg['buffers'].items() if v['role']=='input'},options={k:v for k,v in opts.items() if k!='num_tokens_post_padded_ptr'},kernel_source=dict(path=str(source),sha256=source_hash,commit=commit),frequency_ghz=float(os.environ["GEMM1_GC_GHZ"]),warmup=False,input_kind=kind,output_dtype='torch.bfloat16')
(root/'launch.json').write_text(json.dumps(launch,indent=2))
compiled=mod.gemm_top_w8a8_channel_wasp_persistent[(grid,)](a,b,sa,sb,c,ids,experts,M,E,N,K,topk,*sb.stride(),0,*sa.stride(),**opts)
actual=c.cpu();torch.testing.assert_close(actual[:output_rows],golden[:output_rows],rtol=.02,atol=.002);assert torch.isnan(actual[output_rows:]).all();assert torch.isfinite(actual[:output_rows]).all()
for ext in ('hsaco','amdgcn','llir','ttgir'):
 value=compiled.asm.get(ext)
 if isinstance(value,bytes):(root/f'kernel.{ext}').write_bytes(value)
 elif isinstance(value,str):(root/f'kernel.{ext}').write_text(value)
result=dict(mismatches=0,checked_elements=output_rows*N,untouched_tail=True,input_kind=kind,shared=compiled.metadata.shared,output_sha256=hashlib.sha256(actual[:output_rows].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),kernel_metadata=compiled.metadata._asdict())
(root/'result.json').write_text(json.dumps(result,indent=2,default=str));print({k:v for k,v in result.items() if k!='kernel_metadata'},flush=True)
