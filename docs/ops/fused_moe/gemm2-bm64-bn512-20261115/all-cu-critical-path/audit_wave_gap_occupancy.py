import json,re,sys,collections
from pathlib import Path
root=Path(sys.argv[1]);cu,simd,role,ordinal=map(int,sys.argv[2:]);waves={};events=[]
p=root/f"m5out/0/0/itrace_cu{cu%12}_se{cu//12}_xcd0.txt"
rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: (.*)")
with p.open() as f:
 period=int(re.search(r'GC_CLOCK (\d+)',f.readline())[1])
 for l in f:
  m=rx.match(l)
  if not m:continue
  tick=int(m[1]);w=m[2];op=m[3];v=re.search(r'WAVE_START\[CU(\d+)_SIMD(\d+)_WAVE(\d+)\]',op)
  if v:assert int(v[1])==cu;waves[w]=(int(v[2]),int(v[3]));continue
  if w in waves and waves[w][0]==simd and op.startswith('v_mmac_'):events.append((tick,waves[w][1]))
selected=[t for t,r in events if r==role];begin=selected[ordinal]+8*period;end=selected[ordinal+1]
other=[(t,r) for t,r in events if r!=role and begin<=t<end]
busy=sum(min(end,t+8*period)-max(begin,t) for t,r in other)
result={'cu':cu,'simd':simd,'observed_role':role,'after_mmac_ordinal':ordinal,'begin_tick':begin,'end_tick':end,'period_ticks':period,'per_wave_gap_cycles':(end-begin)/period,'other_role_instruction_counts':dict(collections.Counter(r for t,r in other)),'other_role_busy_cycles':busy/period,'true_idle_cycles':(end-begin-busy)/period}
assert result['per_wave_gap_cycles']==520 and result['other_role_busy_cycles']==512 and result['true_idle_cycles']==8
(root/'wave-gap-occupancy-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
