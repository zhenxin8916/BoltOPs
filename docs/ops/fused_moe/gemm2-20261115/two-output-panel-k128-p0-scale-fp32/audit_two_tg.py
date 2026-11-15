import re,json,sys,collections,hashlib
from pathlib import Path
r=Path(sys.argv[1]);result=json.loads((r/'result.json').read_text());assert result['mismatches']==0
trace=r/'m5out/0/0/itrace_cu0_se0_xcd0.txt';rx=re.compile(r'\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)')
active={};mm=collections.defaultdict(list);events=[];physical={};waits=collections.defaultdict(list);saved={};warm=0
with trace.open() as f:
 period=int(re.search(r'GC_CLOCK (\d+)',f.readline())[1])
 for line in f:
  if not any(x in line for x in ['WAVE_START[',': s_endpgm','ACTIVE:','v_mmac_f32_16x16x32_fp8_fp8','s_abarrier_try_wait ',': r [S']):continue
  m=rx.match(line)
  if not m:continue
  t=int(m[1]);full=int(m[2],16);w=full&65535;physical[w]=full>>16;op=m[4];key=(w,m[3])
  if op.startswith('WAVE_START['):events.append((t,1,w))
  elif op.startswith('s_endpgm'):events.append((t,-1,w))
  elif op.startswith('ACTIVE:'):active[w]=int(op.split()[-1],16)
  elif op.startswith('v_mmac_f32_16x16x32_fp8_fp8'):
   if active.get(w):mm[w].append(t)
   else:warm+=1
  elif op.startswith('s_abarrier_try_wait '):
   e=dict(op=op,reads={});waits[w].append(e);saved[key]=e
  elif key in saved and op.startswith('r [S'):
   q=re.match(r'r \[S(\d+)\]([0-9a-f]+)',op)
   if q:saved[key]['reads'].setdefault(int(q[1]),int(q[2],16))
live=set();peak=0;peak_simd=None
for t,delta,w in sorted(events):
 if delta==1:assert w not in live;live.add(w)
 else:assert w in live;live.remove(w)
 if len(live)>peak:peak=len(live);peak_simd=[sum(physical[w]==i for w in live) for i in range(4)]
assert not live
stats=(r/'m5out/0/0/stats.txt').read_text();model_count=int(re.search(r'^system.shaders.CUs00.numInstrTypeExecuted::MMOP\s+(\d+)',stats,re.M)[1]);assert sum(map(len,mm.values()))==model_count
simds=[]
for simd in range(4):
 times=sorted(t for w,ts in mm.items() if physical[w]==simd for t in ts);last=-1;busy=0
 for t in times:busy+=max(0,t+8*period-max(last,t));last=max(last,t+8*period)
 assert busy==len(times)*8*period,(simd,busy,len(times)*8*period)
 simds.append(dict(simd=simd,mmac=len(times),busy_cycles=busy//period))
audit={}
for w,es in waits.items():
 ids=[]
 for e in es:
  operand=e['op'].split(None,1)[1].split(',')[1].strip();ids.append(e['reads'][int(operand[1:])] if operand.startswith('s') else int(operand))
 assert ids[0]!=0 and all(0<=x<=15 for x in ids) and all(a!=b for a,b in zip(ids,ids[1:])),(w,ids)
 audit[w]=dict(first_id=ids[0],waits=len(ids),all_ids_in_range=True,no_adjacent_repeat=True)
out=dict(peak_live_waves_cu0=peak,per_simd_at_peak=peak_simd,starts=sum(d==1 for _,d,_ in events),ends=sum(d==-1 for _,d,_ in events),cu0_mmac_count=model_count,exec0_mmac_count=warm,mmac_busy8_cycle_union_verified=True,simds=simds,wait_id_audit=audit,numerical=result,definition='actual CU0 wave start/end sweep; all starts and ends included, not a sum across sequential TGs; metrics acceptance separately uses full 48-CU dispatch')
(r/'two-tg-audit.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k] for k in ['peak_live_waves_cu0','per_simd_at_peak','starts','ends','cu0_mmac_count','exec0_mmac_count']})
