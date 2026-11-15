import sys,re,json,collections
from pathlib import Path
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)")
mls=collections.defaultdict(list);gate={};loads=collections.defaultdict(list);starts={};counts={};firstwait={}
for line in (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open():
 if not any(x in line for x in [': matrix_load_',': buffer_load_',': s_abarrier_arrive ',': s_abarrier_init ',': s_abarrier_try_wait ',': WAVE_START']):continue
 m=rx.match(line)
 if not m:continue
 t=int(m[1]);full=int(m[2],16);w=full&65535;op=m[4]
 if op.startswith('WAVE_START'):starts[w]=t
 if op.startswith('matrix_load_'):mls[w].append(t)
 if op.startswith('s_abarrier_init '):
  q=re.match(r's_abarrier_init\s+(\d+), (\d+)',op)
  if q:counts[int(q[1])]=int(q[2])
 if op.startswith('s_abarrier_try_wait '):firstwait.setdefault(w,op)
 if op.startswith('s_abarrier_arrive ') and ', 10,' in op and w<8:
  assert w not in gate;gate[w]=t
 if op.startswith('buffer_load_') and w>=8:loads[w].append(t)
assert set(gate)==set(range(8)),gate
assert counts[10]==8,counts
assert all(', 10, 0' in firstwait[w] for w in range(8,16)),firstwait
producer=[]
for w,t in sorted(gate.items()):
 q=[x for x in mls[w] if x<t];assert len(q)==(8 if w<4 else 16),(w,len(q))
 producer.append(dict(wave=w,role='P0' if w<4 else 'P1',initial_mls=len(q),last_mls_tick=max(q),gate_arrive_tick=t))
latest_mls=max(e['last_mls_tick'] for e in producer);latest_arrive=max(gate.values())
consumer=[]
for w,ts in sorted(loads.items()):
 assert min(ts)>latest_arrive>latest_mls,(w,min(ts),latest_arrive,latest_mls)
 consumer.append(dict(wave=w,role='C0' if w<12 else 'C1',first_buffer_load_tick=min(ts),after_all_initial_producer_mls=True,after_all_initial_gate_arrivals=True))
assert len(consumer)==8
out=dict(producer=producer,consumer=consumer,abarrier10_count=counts[10],definition='CU0 all four SIMD: every initial producer MLS precedes the first consumer metadata buffer load; arrival gates instruction issue only, not memory return. Token10 is reused with count8 and subsequent C1 arrives count2 per wave.')
(r/'scale-issue-gate-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
