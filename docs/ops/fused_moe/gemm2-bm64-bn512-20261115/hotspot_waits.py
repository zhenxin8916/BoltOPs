from pathlib import Path
import re,json,sys,collections
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: (.*)")
ops=collections.defaultdict(list)
with (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open() as f:
 period=int(re.search(r'GC_CLOCK (\d+)',f.readline())[1])
 for line in f:
  m=rx.match(line)
  if not m:continue
  w=int(m[2],16);op=m[3]
  if w>>16 or op.startswith(('r ','w ')):continue
  ops[w].append((int(m[1]),op))
out={}
for w in (0,4,8,12):
 rows=ops[w];waits=[]
 for (t,o),(n,_) in zip(rows,rows[1:]):
  if o.startswith(('s_abarrier_try_wait','s_waitcnt')):
   waits.append(dict(tick=t,op=o,next_instruction_tick=n,issue_gap_cycles=(n-t)/period))
 out[str(w)]={'longest_wait_issue_gaps':sorted(waits,key=lambda x:x['issue_gap_cycles'],reverse=True)[:16]}
(r/'hotspot-waits.json').write_text(json.dumps(out,indent=2)+'\n')
print('Issue gaps include scheduling/resource effects; not pure instruction latency.')
