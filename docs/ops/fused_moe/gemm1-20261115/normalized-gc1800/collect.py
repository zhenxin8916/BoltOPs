from pathlib import Path
import json,re,hashlib,configparser,sys,shutil
from perf_dispatch_span import read_dispatch_span
from boltops.tools.gluon_moe_experiment_csv import upsert_record
root=Path(__file__).parent
work=Path('/zhenxin/worktrees/BoltOPs-gemm1-config-20261115');checkout=Path('/zhenxin/BoltOPs')
config_rel=Path('boltops/fused_moe/gluon')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def publish(stage,record,launch,result,metrics=None):
 target_rel=Path('docs/ops/fused_moe/gemm1-20261115/normalized-gc1800')/record.pop('record_id')
 record['source']=str(target_rel/'record.json')
 record.update(json.loads((root/'compiler-changes.json').read_text())[stage])
 for repo in (work,checkout):
  dest=repo/target_rel;dest.mkdir(parents=True,exist_ok=True)
  for name,x in [('record.json',record),('launch.json',launch),('result.json',result)]:
   (dest/name).write_text(json.dumps(x,indent=2,default=str)+'\n')
  if metrics is not None:(dest/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
  path=upsert_record(stage,record,repo/config_rel)
 print(stage,record['bn'],record['quant_type'],record['time_us'],path,flush=True)

def base_record(launch,stage,bn,bk,kind,source_hash,gc,wave,time):
 case=launch['case'];grid=launch.get('physical_cus',launch.get('active_ctas',48));count=1024//bn
 return dict(arch='gfx946',quant_type='f8_w8a8_channel' if kind=='fp8' else 'int8_w8a8_channel',indtype='torch.bfloat16',token=case['M'],inter_dim=case['N'],model_dim=case['K'],expert=case['E'],topk=case['topk'],q_size_n=0,q_size_k=0,sol_type='triton',sol_id='triton:gluon_moe_gemm1' if stage=='top' else 'triton:gluon_moe_gemm2',time_us=time,wave_span_time_us=wave,gc_ghz=1.8,effective_gc_ghz=gc,cu_count=grid,bm=128,bn=bn,bk=bk,tiles_per_tg=count,total_tiles=grid*count,output_elements_per_tg=128*1024,actual_n=launch['actual_N'],actual_k=launch['actual_K'],time_scope='full_dispatch',work_scope='bounded_equal_padded_output_work',warmup=0,kernel_sha256=source_hash,input_sha256=digest(launch['input_sha256']),source='',verified=True,mismatches=0)

mode=sys.argv[1] if len(sys.argv)>1 else 'top'
if mode=='bottom':
 bases=[('bottom-fp8-bn512','bn512-bk64-best-gc1800',512,64),('bottom-fp8-bn256','bn512-bk64-study/frozen-bn256-four-output-control',256,128)]
 for record_id,rel,bn,bk in bases:
  d=checkout/'docs/ops/fused_moe/gemm2-20261115'/rel
  launch=json.loads((d/'launch.json').read_text());result=json.loads((d/'result.json').read_text());metrics=json.loads((d/'model-metrics.json').read_text())
  assert result['mismatches']==0
  assert launch['persistent_outputs_per_tg']==1024//bn
  assert launch['measured_logical_tiles']==48*1024//bn
  # Resolve the original run directory from the archived dispatch timing evidence.
  original=Path(metrics['dispatch_span']['source_perf']).parents[3]
  config=configparser.ConfigParser(strict=False);config.read(original/'m5out/config.ini')
  assert int(config['system.shaders.clk_domain']['clock'])==556
  bk=launch['options']['BLOCK_SIZE_K']
  assert launch['options']['BLOCK_SIZE_N']==bn
  row=base_record(launch,'bottom',bn,bk,'fp8',launch['kernel_source']['sha256'],1000/556,metrics['wave_span_time_us'],metrics['time_us'])
  row.update(record_id=record_id,evidence=str(d),clock_period_ps=556)
  publish('bottom',row,launch,result,metrics)
else:
 for d in sorted(root.glob('top-*-bn*')):
  if not (d/'exit-code.txt').exists() or not (d/'result.json').exists():print('pending',d.name,flush=True);continue
  assert (d/'exit-code.txt').read_text().strip()=='0',d
  launch=json.loads((d/'launch.json').read_text());result=json.loads((d/'result.json').read_text())
  config=configparser.ConfigParser(strict=False);config.read(d/'m5out/config.ini')
  assert int(config['system.shaders.clk_domain']['clock'])==556
  stats=(d/'m5out/0/0/stats.txt').read_text()
  def stat(k):return float(re.search(r'^'+re.escape(k)+r'\s+(\S+)',stats,re.M)[1])
  groups=[int(float(v)) for v in re.findall(r'^system\.shaders\.CUs\d+\.completedWGs\s+(\S+)',stats,re.M)]
  assert len(groups)==48 and set(groups)=={1},groups
  metrics=read_dispatch_span(d);wave=(stat('system.shaders.lastWaveEndTick')-stat('system.shaders.firstWaveStartTick'))/1e6
  opts=launch['options'];bn=opts['BLOCK_SIZE_N'];kind=launch['input_kind']
  assert launch['persistent_outputs_per_tg']==1024//bn and launch['measured_logical_tiles']==48*1024//bn
  assert result['mismatches']==0 and result['untouched_tail'] and result['checked_elements']==48*128*1024
  row=base_record(launch,'top',bn,128,kind,launch['kernel_source']['sha256'],stat('gcFrequency'),wave,metrics['duration_us'])
  row.update(record_id=d.name,evidence=str(d),clock_period_ps=556,output_sha256=result['output_sha256'])
  publish('top',row,launch,result,metrics)
