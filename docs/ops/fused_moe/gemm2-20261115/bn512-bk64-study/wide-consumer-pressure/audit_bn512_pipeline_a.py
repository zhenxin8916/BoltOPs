import re,json,sys,collections
from pathlib import Path
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: (.*)")
counts=collections.defaultdict(int);events=collections.defaultdict(list);lastwait=collections.defaultdict(lambda:None)
for line in (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open():
 m=rx.match(line)
 if not m:continue
 w=int(m[2],16);op=m[3]
 if (w&65535)<8:continue
 if op.startswith('v_mmac_f32_16x16x32_fp8_fp8'):counts[w]+=1
 if op.startswith('s_waitcnt lgkmcnt(0)'):lastwait[w]=int(m[1])
 if op.startswith('s_abarrier_arrive ') and op.split(None,1)[1].split(',')[1].strip()=='3':
  assert lastwait[w] is not None
  events[w].append(dict(tick=int(m[1]),prior_lgkm0_tick=lastwait[w],mmac_before_release=counts[w]))
assert len(events)==8,len(events)
for w,es in events.items():
 assert [e['mmac_before_release'] for e in es]==[0,64,128,192,256,320,384,448],(w,es)
 assert all(e['prior_lgkm0_tick']<e['tick'] for e in es)
(r/'pipeline-a-lifetime-audit.json').write_text(json.dumps({f'{w:08x}':es for w,es in events.items()},indent=2)+'\n')
print('8 consumers: A credit returned once per K64 capture and lgkm0; 0/64/128/192 per M prior MMAC')
