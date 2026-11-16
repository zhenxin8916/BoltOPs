import re,json,sys,collections
from pathlib import Path
root=Path(sys.argv[1]);out=[];rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: (.*)")
for trace in sorted((root/'m5out/0/0').glob('itrace_cu*_se*_xcd0.txt')):
 waves={};rows=collections.defaultdict(list);ordinals=collections.Counter();cu=None
 with trace.open() as f:
  period=int(re.search(r'GC_CLOCK (\d+)',f.readline())[1])
  for l in f:
   m=rx.match(l)
   if not m:continue
   t=int(m[1]);w=m[2];o=m[3];v=re.search(r'WAVE_START\[CU(\d+)_SIMD(\d+)_WAVE(\d+)\]',o)
   if v:cu=int(v[1]);waves[w]=(int(v[2]),int(v[3]));continue
   if w not in waves or not o.startswith('v_mmac_'):continue
   simd,role=waves[w];assert role in [2,3]
   rows[simd].append((t,role,ordinals[w]));ordinals[w]+=1
 assert len(rows)==4 and len(ordinals)==8 and all(x==256 for x in ordinals.values())
 for simd,events in rows.items():
  events.sort();gaps=[]
  for a,b in zip(events,events[1:]):
   delta=b[0]-a[0]-8*period;assert delta>=0,(cu,simd,a,b,delta)
   if delta:gaps.append({'begin_tick':a[0]+8*period,'end_tick':b[0],'cycles':delta/period,'before_role':a[1],'before_ordinal':a[2],'after_role':b[1],'after_ordinal':b[2]})
  span=events[-1][0]+8*period-events[0][0];busy=len(events)*8*period
  assert busy+sum(x['end_tick']-x['begin_tick'] for x in gaps)==span
  out.append({'cu':cu,'simd':simd,'mmac_count':len(events),'first_tick':events[0][0],'last_end_tick':events[-1][0]+8*period,'busy_cycles':len(events)*8,'mmac_span_cycles':span/period,'mmac_span_utilization_percent':100*busy/span,'largest_true_idle_gaps':sorted(gaps,key=lambda x:x['cycles'],reverse=True)[:8]})
assert len(out)==192 and sum(x['mmac_count'] for x in out)==98304
(root/'all-cu-mmac-union.json').write_text(json.dumps({'definition':'Union of C0/C1 MMAC issue+8-cycle intervals per physical SIMD, all 48 CU x 4 SIMD; MMAC-span utilization is not the full-dispatch acceptance denominator. No overlapping intervals and all 98304 expected instructions verified.','simds':out},indent=2)+'\n')
print('192 SIMD union audits passed, 98304 MMAC')
for x in out:
 if x['cu']==25 and x['simd']==0:print(x)
