import sys,re,json,collections
from pathlib import Path
r=Path(sys.argv[1]); seq=collections.defaultdict(list);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)"); keys={};starts={}
with (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open() as f:
 period=int(re.search(r'GC_CLOCK (\d+)',f.readline())[1])
 for line in f:
  m=rx.match(line)
  if not m:continue
  t,w,i,o=int(m[1]),int(m[2],16),m[3],m[4]
  if w not in (0,4,8,12):continue
  if o.startswith('WAVE_START'):starts[w]=t
  if re.match(r'(s_|v_|ds_|buffer_|global_|matrix_)',o):
   e=dict(t=t,op=o,reads={});seq[w].append(e);keys[w,i]=e
  elif o.startswith('r [S') and (w,i) in keys:
   q=re.match(r'r \[S(\d+)\]([0-9a-f]+)',o)
   if q:keys[w,i]['reads'][int(q[1])]=int(q[2],16)
out={}
for w in (8,12):
 ops=seq[w];ix=[i for i,e in enumerate(ops) if e['op'].startswith('v_mmac_')];assert len(ix)==128
 a,z=ix[63],ix[64];sub=ops[a+1:z];stores=[e for e in sub if e['op'].startswith('buffer_store')];end=ops[a]['t']+8*period
 waits=[]
 for j in range(a+1,z):
  e=ops[j]
  if e['op'].startswith(('s_abarrier_try_wait','s_waitcnt')):
   d=dict(op=e['op'],tick=e['t'],next_issue_gap_cycles=(ops[j+1]['t']-e['t'])/period)
   if e['op'].startswith('s_abarrier_try_wait'):
    operand=e['op'].split(',')[1].strip();d['id']=e['reads'].get(int(operand[1:])) if operand.startswith('s') else int(operand)
   waits.append(d)
 out['C0' if w==8 else 'C1']=dict(first_mmac_tick=ops[ix[0]]['t'],initial_to_mmac_cycles=(ops[ix[0]]['t']-starts[w])/period,inter_output_gap_cycles=(ops[z]['t']-end)/period,mmac_end_to_first_store_cycles=(stores[0]['t']-end)/period,first_to_last_store_issue_cycles=(stores[-1]['t']-stores[0]['t'])/period,last_store_to_next_mmac_cycles=(ops[z]['t']-stores[-1]['t'])/period,waits=waits,last_mmac_end_tick=ops[ix[-1]]['t']+8*period,tail_after_mmac_cycles=(ops[-1]['t']-ops[ix[-1]]['t']-8*period)/period)
(r/'phase-analysis.json').write_text(json.dumps(out,indent=2));print(r.name,json.dumps(out))
