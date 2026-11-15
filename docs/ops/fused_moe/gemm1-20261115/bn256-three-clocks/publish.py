from pathlib import Path
import json,shutil,configparser,re,hashlib,subprocess
from boltops.tools.gluon_moe_experiment_csv import upsert_record
root=Path(__file__).parent;repo=Path('/zhenxin/BoltOPs')
rel=Path('docs/ops/fused_moe/gemm1-20261115/bn256-three-clocks');dest=repo/rel
oldpath=repo/'docs/ops/fused_moe/gemm1-20261115/release-5b24821-gc1800/top-fp8-bn256'
old=json.loads((oldpath/'record.json').read_text());old_launch=json.loads((oldpath/'launch.json').read_text());old_result=json.loads((oldpath/'result.json').read_text())
triton='/zhenxin/worktrees/triton-feature-20261102';git=lambda *args:subprocess.check_output(['git','-C',triton,*args],text=True).strip()
assert git('rev-parse','HEAD')==old['triton_commit'];assert not git('status','--porcelain')
summary=[]
for gc in (2200,1800,1500):
 d=root/f'baseline-gc{gc}';launch=json.loads((d/'launch.json').read_text());result=json.loads((d/'result.json').read_text());metrics=json.loads((d/'metrics.json').read_text())
 assert result['mismatches']==0 and result['output_sha256']==old_result['output_sha256']
 assert launch['input_sha256']==old_launch['input_sha256'];assert launch['kernel_source']['sha256']==old['kernel_sha256']
 stats=(d/'m5out/0/0/stats.txt').read_text();groups=[int(float(v)) for v in re.findall(r'^system\.shaders\.CUs\d+\.completedWGs\s+(\S+)',stats,re.M)]
 assert len(groups)==48 and set(groups)=={1}
 row=dict(old);row.update({k:metrics[k] for k in ('time_us','wave_span_time_us','mmac_total_pct','mmac_scope','effective_gc_ghz')});row.update(gc_ghz=gc/1000,source=str(rel/f'gc{gc}/record.json'),evidence=str(d),clock_period_ps=metrics['clock_period_ps'])
 out=dest/f'gc{gc}';out.mkdir(parents=True,exist_ok=True)
 for name,data in [('record.json',row),('launch.json',launch),('result.json',result),('metrics.json',metrics)]:
  (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
 upsert_record('top',row)
 summary.append(dict(gc_ghz=gc/1000,bm=128,bn=256,bk=128,time_us=metrics['time_us'],mmac_total_pct=metrics['mmac_total_pct'],mmac_scope=metrics['mmac_scope'],checked_elements=result['checked_elements'],mismatches=0))
for file in ('machine-code-comparison.json','source-reference.json','baseline.py','probe.py','run.py','analyze.py','perf_dispatch_span.py','publish.py'):
 shutil.copy2(root/file,dest/file)
shutil.copy2('/zhenxin/temp/moe-release-5b24821-fp8-20261115/toolchain.json',dest/'toolchain.json')
(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
