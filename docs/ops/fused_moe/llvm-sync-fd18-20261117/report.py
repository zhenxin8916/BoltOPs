import collections, gzip, hashlib, json, re, shutil, struct, sys
from pathlib import Path
root=Path('/zhenxin/temp/llvm-sync-fd18-20261117')
repo=Path('/zhenxin/BoltOPs')
doc=repo/'docs/ops/fused_moe/llvm-sync-fd18-20261117'
doc.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(repo/'docs/ops/fused_moe/gemm1-bm32-bn256-integration-20261117'))
from perf_dispatch_span import read_dispatch_span

def text_bytes(path):
 b=path.read_bytes();e=struct.unpack_from('<16sHHIQQQIHHHHHH',b); off,size,count,strings=e[6],e[11],e[12],e[13]
 sections=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*size) for i in range(count)]
 s=sections[strings]; names=b[s[4]:s[4]+s[5]]
 for s in sections:
  name=names[s[0]:].split(b'\0')[0]
  if name==b'.text':return b[s[4]:s[4]+s[5]]
 raise ValueError(path)
def record(d):
 p=d/'gemm/kernel.amdgcn';s=p.read_text()
 return dict(time_us=read_dispatch_span(d)['duration_us'],ds_counts=dict(collections.Counter(re.findall(r'^\s+(ds_\w+)',s,re.M))),mls_counts=dict(collections.Counter(re.findall(r'^\s+(matrix_load_\w+)',s,re.M))),text_sha256=hashlib.sha256(text_bytes(d/'gemm/kernel.hsaco')).hexdigest(),valu_count=len(re.findall(r'^\s+v_\w+',s,re.M)),salu_count=len(re.findall(r'^\s+s_\w+',s,re.M)),spill_fields=re.findall(r'(?:\.sgpr_spill_count|\.vgpr_spill_count|\.private_segment_fixed_size):\s*(\d+)',s),result=json.loads((d/'result.json').read_text()))
def archive(src,dst):
 dst.mkdir(parents=True,exist_ok=True)
 for name in ['launch.json','result.json','exit-code.txt']:
  if (src/name).exists():shutil.copy2(src/name,dst/name)
 for name in ['kernel.amdgcn','kernel.llir','kernel.ttgir','kernel.source']:
  p=src/'gemm'/name
  if p.exists():
   with gzip.open(dst/(name+'.gz'),'wb') as f:f.write(p.read_bytes())
 for name in ['kernel.hsaco','compiled.json']:
  p=src/'gemm'/name
  if p.exists():shutil.copy2(p,dst/name)
 for p in (src/'m5out/0/0').glob('*.perf'):shutil.copy2(p,dst/p.name)
 with gzip.open(dst/'run.log.gz','wb') as f:f.write((src/'run.log').read_bytes())
rows=[]
for stage,bn in [('top',128),('top',256),('top',512),('bottom',256),('bottom',512)]:
 paths={v:root/'bm128'/f'{v}-{stage}-bn{bn}' for v in ['old','new']}
 if not all((d/'exit-code.txt').exists() for d in paths.values()):continue
 if (paths['new']/'exit-code.txt').read_text().strip()!='0':
  failure_log=(paths['new']/'run.log').read_text(errors='replace')
  if 'ValueError: WASP partition quotas exceed the per-SIMD VGPR budget' not in failure_log or (paths['new']/'gemm/kernel.hsaco').exists() or (paths['new']/'result.json').exists():raise RuntimeError('Unexpected failure; do not classify as quota rejection')
  if (paths['old']/'exit-code.txt').read_text().strip()!='0':raise RuntimeError('old baseline failed')
  row=dict(case=f'gemm{1 if stage=="top" else 2}-bm128-bn{bn}',old=record(paths['old']),new=dict(exit_code=int((paths['new']/'exit-code.txt').read_text()),compile_failed=True,reason='WASP partition quotas exceed the per-SIMD VGPR budget',time_us=None),text_identical=None,ds_identical=None,output_identical=None,time_change_pct=None)
  rows.append(row);archive(paths['old'],doc/'bm128'/row['case']/'old')
  dst=doc/'bm128'/row['case']/'new';dst.mkdir(parents=True,exist_ok=True)
  shutil.copy2(paths['new']/'exit-code.txt',dst/'exit-code.txt')
  with gzip.open(dst/'run.log.gz','wb') as f:f.write((paths['new']/'run.log').read_bytes())
  print(row['case'],'NEW COMPILE FAILED',flush=True);continue
 pair={v:record(d) for v,d in paths.items()}
 a,b=pair['old'],pair['new']
 row=dict(case=f'gemm{1 if stage=="top" else 2}-bm128-bn{bn}',old=a,new=b,llvm_ir_identical=(paths['old']/'gemm/kernel.llir').read_bytes()==(paths['new']/'gemm/kernel.llir').read_bytes(),time_change_pct=(b['time_us']/a['time_us']-1)*100,text_identical=a['text_sha256']==b['text_sha256'],ds_identical=a['ds_counts']==b['ds_counts'],output_identical=a['result']['output_sha256']==b['result']['output_sha256'])
 rows.append(row)
 for v,d in paths.items():archive(d,doc/'bm128'/row['case']/v)
 print(row['case'],a['time_us'],b['time_us'],row['time_change_pct'],row['text_identical'],flush=True)
(doc/'bm128-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
probes=doc/'ds-probes';probes.mkdir(exist_ok=True)
for p in (root/'probes').iterdir():
 if p.is_file():
  if p.suffix in ('.s','.mir'):
   with gzip.open(probes/(p.name+'.gz'),'wb') as f:f.write(p.read_bytes())
  else:shutil.copy2(p,probes/p.name)
for name in ['status-before.txt','log-before.txt','old-binaries.sha256','new-binaries.sha256','fd18.patch','build.log','run-bm128.py','run-bm64g1.py']:
 if (root/name).exists():
  if name in ('fd18.patch','log-before.txt'):
   with gzip.open(doc/(name+'.gz'),'wb') as f:f.write((root/name).read_bytes())
  else:shutil.copy2(root/name,doc/name)
