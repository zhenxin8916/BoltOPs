from pathlib import Path
import sys,re,json,collections
r=Path(sys.argv[1]);seq=collections.defaultdict(list);starts={};rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)")
with (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open() as f:
 period=int(re.search(r'GC_CLOCK (\d+)',f.readline())[1])
 for line in f:
  m=rx.match(line)
  if not m:continue
  t,w,o=int(m[1]),int(m[2],16),m[4]
  if w not in (0,4,8,12):continue
  if o.startswith('WAVE_START'):starts[w]=t
  if re.match(r'(s_|v_|ds_|buffer_|global_|matrix_)',o):seq[w].append(dict(t=t,op=o))
out={}
for w in (8,12):
 ops=seq[w];ix=[i for i,e in enumerate(ops) if e['op'].startswith('v_mmac_')];assert len(ix)==128
 a,z=ix[63],ix[64];st=[e for e in ops[ix[-1]+1:] if e['op'].startswith('buffer_store')]
 out[str(w)]=dict(first_mmac_tick=ops[ix[0]]['t'],last_mmac_end_tick=ops[ix[-1]]['t']+8*period,first_wave_to_mmac_cycles=(ops[ix[0]]['t']-starts[w])/period,k128_boundary_gap_cycles=(ops[z]['t']-ops[a]['t'])/period-8,first_to_last_store_issue_cycles=(st[-1]['t']-st[0]['t'])/period,last_mmac_end_to_first_store_cycles=(st[0]['t']-ops[ix[-1]]['t'])/period-8,tail_after_last_mmac_cycles=(ops[-1]['t']-ops[ix[-1]]['t'])/period-8,boundary_ops=[dict(op=ops[i]['op'],next_issue_cycles=(ops[i+1]['t']-ops[i]['t'])/period) for i in range(a+1,z)])
(r/'pair-phase-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(r.name,json.dumps(out))
