import sys,re,json,collections
from pathlib import Path
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)")
pending={};saved={};waits=[];hist=collections.Counter();period=None
with (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open() as f:
 period=int(re.search(r'GC_CLOCK (\d+)',f.readline())[1])
 for line in f:
  if not any(x in line for x in [': s_',': v_',': ds_',': buffer_',': global_',': matrix_',': r [S']):continue
  m=rx.match(line)
  if not m:continue
  t=int(m[1]);wave=int(m[2],16);op=m[4]
  if op.startswith('r [S'):
   e=saved.get((wave,m[3]));q=re.match(r'r \[S(\d+)\]([0-9a-f]+)',op)
   if e is not None and q and e['operand'].startswith('s') and int(e['operand'][1:])==int(q[1]):e['id']=int(q[2],16)
   continue
  hist[op.split()[0]]+=1
  if wave in pending:
   e=pending.pop(wave);e['cycles']=(t-e.pop('tick'))/period;waits.append(e)
  if op.startswith('s_abarrier_try_wait '):
   operand=op.split(None,1)[1].split(',')[1].strip();e=dict(wave=wave,tick=t,operand=operand)
   if not operand.startswith('s'):e['id']=int(operand)
   pending[wave]=e;saved[(wave,m[3])]=e
summary={}
for id in sorted(set(e['id'] for e in waits)):
 es=[e['cycles'] for e in waits if e['id']==id];summary[id]=dict(count=len(es),sum_cycles=sum(es),max_cycles=max(es))
out=dict(wait_to_next_instruction_cycles=summary,stores={k:v for k,v in hist.items() if 'store' in k},definition='CU0 all waves; wait issue to next instruction issue, includes execution overhead, not pure ready latency')
(r/'n-pipeline-waits.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
