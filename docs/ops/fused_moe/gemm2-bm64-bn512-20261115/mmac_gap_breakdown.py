from pathlib import Path
import re,json,sys,collections
r=Path(sys.argv[1]);rx=re.compile(r'\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: ((?:s_|v_|ds_|buffer_|global_|matrix_).*)');seq=collections.defaultdict(list)
f=r/'m5out/0/0/itrace_cu0_se0_xcd0.txt'
with f.open() as fp:
 period=int(re.search(r'GC_CLOCK (\d+)',fp.readline())[1])
 for line in fp:
  if ': ' not in line:continue
  m=rx.match(line)
  if not m:continue
  w=int(m[2],16)
  if w>>16!=0 or not 8<=(w&65535)<16:continue
  seq[w].append((int(m[1]),m[3]))
out={}
for w,ops in seq.items():
 idx=[i for i,(t,o) in enumerate(ops) if o.startswith('v_mmac_')]
 assert len(idx)==256,(w,len(idx))
 gaps=[]
 for ordinal,(a,z) in enumerate(zip(idx,idx[1:])):
  cyc=(ops[z][0]-ops[a][0])/period-8
  if cyc>0:
   sub=ops[a+1:z];hist=collections.Counter(o.split()[0] for t,o in sub)
   gaps.append(dict(after_mmac_ordinal=ordinal,gap_cycles=cyc,first_tick=ops[a][0]+8*period,next_tick=ops[z][0],instruction_counts=dict(hist),first_ops=[o for t,o in sub[:8]],last_ops=[o for t,o in sub[-8:]]))
 out[str(w)]=dict(role='C0' if (w&65535)<12 else 'C1',mmac_first_tick=ops[idx[0]][0],mmac_last_end_tick=ops[idx[-1]][0]+8*period,top_gaps=sorted(gaps,key=lambda x:x['gap_cycles'],reverse=True)[:12],gaps_sum_cycles=sum(g['gap_cycles'] for g in gaps))
(r/'mmac-gap-breakdown.json').write_text(json.dumps(out,indent=2)+'\n')
for d in out.values():print(d['role'],[(x['after_mmac_ordinal'],x['gap_cycles'],x['instruction_counts']) for x in d['top_gaps'][:5]])
