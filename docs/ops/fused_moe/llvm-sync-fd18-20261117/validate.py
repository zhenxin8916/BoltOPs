import hashlib,json,subprocess
from pathlib import Path
r=Path('/zhenxin/temp/llvm-sync-fd18-20261117');bins={'old':r/'old-bin','new':Path('/zhenxin/llvm-project/build-dtk-llvm-6.3.3/bin')}
rows=[]
for version,bin in bins.items():
 for test in ['lds-common-base','lds-wrapped-constant-read2']:
  for mode,suffix in [('IR','opt.ll'),('GCN','s')]:
   inp=r/'probes'/f'{test}.{version}.{suffix}'
   command=[str(bin/'FileCheck'),str(r/'probes'/f'{test}.ll'),f'--check-prefix={mode}']
   result=subprocess.run(command,input=inp.read_bytes(),capture_output=True)
   rows.append(dict(version=version,test=test,mode=mode,exit_code=result.returncode,command=command))
for mode in ['DEFAULT','CONTROL']:
 command=[str(bins['new']/'FileCheck'),'/zhenxin/llvm-project/llvm/test/CodeGen/AMDGPU/hcu-control-only-ebarrier-sync.mir',f'--check-prefix={mode}']
 result=subprocess.run(command,input=(r/'probes'/f'ebarrier-{mode}.mir').read_bytes(),capture_output=True)
 rows.append(dict(version='new',test='control-only-ebarrier-sync',mode=mode,exit_code=result.returncode,command=command))
(r/'validation.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:print(row['version'],row['test'],row['mode'],row['exit_code'])
