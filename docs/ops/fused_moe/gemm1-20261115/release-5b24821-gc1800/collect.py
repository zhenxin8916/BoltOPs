from pathlib import Path
import json,re,configparser,hashlib,shutil,csv
from perf_dispatch_span import read_dispatch_span
from boltops.tools.gluon_moe_experiment_csv import upsert_record
root=Path(__file__).parent;repo=Path('/zhenxin/BoltOPs')
evidence_rel=Path('docs/ops/fused_moe/gemm1-20261115/release-5b24821-gc1800')
evidence=repo/evidence_rel;metadata=json.loads((root/'toolchain.json').read_text());assert metadata['triton_changes']=='无本地改动' and not metadata['triton_status']
summary=[]
for name in ('top-fp8-bn128','top-fp8-bn256'):
 d=root/name;assert (d/'exit-code.txt').read_text().strip()=='0',name
 result=json.loads((d/'result.json').read_text());assert result['mismatches']==0
 launch=json.loads((d/'launch.json').read_text());bn=launch['options']['BLOCK_SIZE_N'];stage=name.split('-')[0]
 old=json.loads((repo/'docs/ops/fused_moe/gemm1-20261115/normalized-gc1800'/name/'record.json').read_text())
 oldlaunch=json.loads((repo/'docs/ops/fused_moe/gemm1-20261115/normalized-gc1800'/name/'launch.json').read_text())
 assert launch['input_sha256']==oldlaunch['input_sha256'],name
 stats=(d/'m5out/0/0/stats.txt').read_text()
 stat=lambda k:float(re.search(r'^'+re.escape(k)+r'\s+(\S+)',stats,re.M)[1])
 groups=[int(float(v)) for v in re.findall(r'^system\.shaders\.CUs\d+\.completedWGs\s+(\S+)',stats,re.M)]
 assert len(groups)==48 and set(groups)=={1},groups
 config=configparser.ConfigParser(strict=False);config.read(d/'m5out/config.ini');assert int(config['system.shaders.clk_domain']['clock'])==556
 span=read_dispatch_span(d);wave=(stat('system.shaders.lastWaveEndTick')-stat('system.shaders.firstWaveStartTick'))/1e6
 row=dict(old);row.update(time_us=span['duration_us'],wave_span_time_us=wave,triton_commit=metadata['triton_commit'],triton_changes='无本地改动',kernel_sha256=launch['kernel_source']['sha256'],source=str(evidence_rel/name/'record.json'),evidence=str(d))
 row['llvm_project_commit']=metadata['llvm_project_commit']
 row.update(verified=True,mismatches=0)
 if stage=='top':assert result['untouched_tail'] and result['checked_elements']==48*128*1024
 out=evidence/name;out.mkdir(parents=True,exist_ok=True)
 for file,data in [('record.json',row),('launch.json',launch),('result.json',result),('metrics.json',dict(dispatch_span=span,time_us=row['time_us'],wave_span_time_us=wave))]:
  (out/file).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
 upsert_record(stage,row,root/'csv')
 summary.append(dict(case=name,bm=row['bm'],bn=bn,bk=row['bk'],previous_us=old['time_us'],time_us=row['time_us'],change_percent=(row['time_us']/old['time_us']-1)*100,checked_elements=result['checked_elements'],mismatches=0,frozen_inputs_identical=True))
for filename in ('tuned_experiment_fmoe_gluon_w8a8channel_top_fp8.csv',):
 shutil.copy2(root/'csv'/filename,repo/'boltops/fused_moe/gluon'/filename)
for filename in ('toolchain.json','machine-code-comparison.json','build.log','collect.py','launch.py','probe_top.py','perf_dispatch_span.py'):
 shutil.copy2(root/filename,evidence/filename)
(evidence/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');print(json.dumps(summary,indent=2,ensure_ascii=False))
