from pathlib import Path
import re,json,collections,configparser,sys
from perf_dispatch_span import read_dispatch_span
root=Path(__file__).parent
rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)")
def events(path, starts, ends):
 current={}
 for line in path.open():
  m=rx.match(line)
  if not m:continue
  tick,wid,iid,op=m.groups();w=int(wid,16)
  if op.startswith('WAVE_START['):starts.append(int(tick))
  elif op.startswith('s_endpgm'):ends.append(int(tick))
  if 'PC: 0x' in op:
   if w in current:yield w,current[w]
   current[w]={'tick':int(tick),'op':'','active':0}
  elif w in current:
   if op.startswith('ACTIVE:'):current[w]['active']=int(op.split()[-1],16)
   elif op.startswith(('v_','s_','ds_','buffer_','matrix_','flat_','global_','scratch_')):current[w]['op']=op
 for w,item in current.items():yield w,item

def union(spans):
 out=[]
 for a,b in sorted(spans):
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out
for name in sys.argv[1:]:
 d=root/name;assert (d/'exit-code.txt').read_text().strip()=='0'
 result=json.loads((d/'result.json').read_text());assert result['mismatches']==0 and result['untouched_tail'] and result['checked_elements']==6291456
 cp=configparser.ConfigParser(strict=False);cp.read(d/'m5out/config.ini');period=int(cp['system.shaders.clk_domain']['clock'])
 dispatch=read_dispatch_span(d);span=dispatch['duration_ticks'];start=dispatch['begin_tick'];end=dispatch['end_tick']
 stats=(d/'m5out/0/0/stats.txt').read_text();stat=lambda key:float(re.search(r'^'+re.escape(key)+r'\s+(\S+)',stats,re.M)[1])
 spans=collections.defaultdict(list);counts=collections.Counter();families=collections.Counter();wave_starts=[];wave_ends=[]
 for w,e in events(d/'m5out/0/0/itrace_cu0_se0_xcd0.txt', wave_starts, wave_ends):
  if e['op'].startswith('v_mmac') and e['active']:
   families[e['op'].split()[0]]+=1
   if e['op'].startswith('v_mmac_f32_16x16x32_fp8_fp8'):
    t=e['tick']; assert start<=t<end
    spans[w>>16].append([t,min(t+8*period,end)]);counts[str(w)]+=1
 assert len(counts)==8 and set(counts.values())=={4096},counts
 unions=[union(spans[i]) for i in range(4)]
 busy=sum(sum(b-a for a,b in xs) for xs in unions)
 assert len(wave_starts)==len(wave_ends)==16
 cu0_first=min(wave_starts);cu0_last=max(wave_ends)+period;cu0_span=cu0_last-cu0_first
 metrics=dict(cu0_lifetime_us=cu0_span/1e6,cu0_first_wave_start_tick=cu0_first,cu0_last_wave_end_tick=cu0_last,dispatch_window_mmac_pct=100*busy/(4*span),time_us=dispatch['duration_us'],wave_span_time_us=(stat('system.shaders.lastWaveEndTick')-stat('system.shaders.firstWaveStartTick'))/1e6,mmac_total_pct=100*busy/(4*cu0_span),mmac_scope='cu0_wave_lifetime',frequency_ghz=int(name.split('gc')[1])/1000,effective_gc_ghz=stat('gcFrequency'),clock_period_ps=period,fp8_busy_ticks=busy,dispatch=dispatch,active_mmac_families=dict(families),fp8_wave_counts=dict(counts),busy_union_matches_sum=(busy==sum(counts.values())*8*period),simd_busy_ticks=[sum(b-a for a,b in xs) for xs in unions],definition='CU0 four-SIMD average; active FP8 MMAC 8-cycle intervals union / CU0 first WAVE_START to last s_endpgm plus one cycle; includes CU0 startup and tail, excludes BF16 accumulator clear and EXEC0 warmup')
 (d/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(name,metrics['time_us'],metrics['mmac_total_pct'],metrics['busy_union_matches_sum'],flush=True)
