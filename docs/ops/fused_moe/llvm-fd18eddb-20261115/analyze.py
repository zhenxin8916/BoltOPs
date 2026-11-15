"""Use only CU0's own lifetime, excluding EXEC0 and BF16 initialization."""
import collections,configparser,hashlib,json,re,sys
from pathlib import Path
from perf_dispatch_span import read_dispatch_span
RX=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)")
def union(intervals):
 out=[]
 for a,b in sorted(intervals):
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out
def analyze(d):
 d=Path(d)
 assert (d/'exit-code.txt').read_text().strip()=='0'
 launch=json.loads((d/'launch.json').read_text());result=json.loads((d/'result.json').read_text())
 assert result['verified'] and result['mismatches']==0 and result['untouched_tail']
 cfg=configparser.ConfigParser(strict=False);cfg.read(d/'m5out/config.ini');period=int(cfg[cfg['system.shaders']['clk_domain']]['clock'])
 stats=(d/'m5out/0/0/stats.txt').read_text()
 def stat(k):return float(re.search(r'^'+re.escape(k)+r'\s+(\S+)',stats,re.M)[1])
 groups=[int(float(x)) for x in re.findall(r'^system.shaders.CUs\d+.completedWGs\s+(\S+)',stats,re.M)]
 assert len(groups)==48 and set(groups)=={1} and stat('system.shaders.dispatcher.numKernelLaunched')==1
 traces=list((d/'m5out/0/0').glob('itrace*.txt'));assert len(traces)==1 and traces[0].name=='itrace_cu0_se0_xcd0.txt'
 active={};intervals=collections.defaultdict(list);per_wave=collections.defaultdict(list);starts=[];ends=[];families=collections.Counter();inactive=collections.Counter();waits=collections.defaultdict(list);saved={}
 family='v_mmac_i32_16x16x32_i8' if launch['kind']=='int8' else 'v_mmac_f32_16x16x32_fp8_fp8'
 for line in traces[0].open():
  if not any(x in line for x in ('ACTIVE:',': v_mmac',': s_abarrier_try_wait ',': r [S','WAVE_START[',': s_endpgm')):continue
  m=RX.match(line)
  if not m:continue
  tick,wid,iid,op=m.groups();tick=int(tick);w=int(wid,16);key=(w,iid)
  if op.startswith('WAVE_START['):starts.append(tick)
  elif op.startswith('s_endpgm'):ends.append(tick)
  elif op.startswith('ACTIVE:'):active[w]=int(op.split()[-1],16)
  elif op.startswith('v_mmac'):
   name=op.split()[0]
   if not active.get(w):inactive[name]+=1;continue
   families[name]+=1
   if name==family:intervals[w>>16].append((tick,tick+8*period));per_wave[w].append(tick)
   else:assert name=='v_mmac_f32_16x16x16_bf16',name
  elif op.startswith('s_abarrier_try_wait '):
   e=dict(op=op,reads={});saved[key]=e;waits[w].append(e)
  elif key in saved and op.startswith('r [S'):
   read=re.match(r'r \[S(\d+)\]([0-9a-f]+)',op)
   if read:saved[key]['reads'].setdefault(int(read[1]),int(read[2],16))
 expected=32768 if launch['stage']=='top' else 4096
 assert len(per_wave)==8 and set(map(len,per_wave.values()))=={expected//8}
 assert len(starts)==len(ends)==16 and len(waits)==16
 ids={}
 for w,rows in waits.items():
  values=[]
  for row in rows:
   token=row['op'].split(None,1)[1].split(',')[1].strip();values.append(row['reads'][int(token[1:])] if token.startswith('s') else int(token))
  assert values[0]!=0 and all(0<=v<16 for v in values) and all(a!=b for a,b in zip(values,values[1:])),(w,values)
  ids[str(w)]=values
 busy=[sum(b-a for a,b in union(intervals[i])) for i in range(4)];assert sum(busy)==expected*8*period
 start=min(starts);end=max(ends)+period;span=end-start
 per_output=expected//8//launch['tiles_per_tg'];hot=[]
 for output in range(launch['tiles_per_tg']):
  for simd in range(4):
   ticks=[t for w,rows in per_wave.items() if w>>16==simd for t in rows[output*per_output:(output+1)*per_output]]
   assert len(ticks)==2*per_output
   hot.append(100*sum(b-a for a,b in union([(t,t+8*period) for t in ticks]))/(max(ticks)+8*period-min(ticks)))
 dispatch=read_dispatch_span(d)
 result=dict(time_us=dispatch['duration_us'],wave_span_time_us=(stat('system.shaders.lastWaveEndTick')-stat('system.shaders.firstWaveStartTick'))/1e6,
  mmac_total_pct=100*sum(busy)/(4*span),hot_loop_mmac_pct=sum(hot)/len(hot),mmac_scope='cu0_wave_lifetime',
  cu0_first_wave_start_tick=start,cu0_last_wave_end_tick=end,cu0_lifetime_us=span/1e6,clock_period_ps=period,effective_gc_ghz=stat('gcFrequency'),gc_ghz=1000/period,
  active_mmac_families=dict(families),inactive_mmac_families=dict(inactive),active_wave_counts={str(w):len(x) for w,x in per_wave.items()},
  abarrier_wait_ids=ids,hardware_wait_rules_passed=True,simd_busy_ticks=busy,dispatch=dispatch,
  hsaco_sha256=hashlib.sha256((d/'gemm/kernel.hsaco').read_bytes()).hexdigest())
 (d/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(d.name,result['time_us'],result['mmac_total_pct'],flush=True)
 return result
if __name__=='__main__':
 for p in sys.argv[1:]:analyze(Path(p))
