from pathlib import Path
import re,json,sys,collections
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) ([0-9a-f]{8}): (.*)");pc={};seq=collections.defaultdict(list)
with (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open() as f:
 period=int(re.search(r'GC_CLOCK (\d+)',f.readline())[1])
 for line in f:
  m=rx.match(line)
  if not m:continue
  t,w,ix,o=int(m[1]),int(m[2],16),m[3],m[4]
  if not 8 <= (w&65535)<16:continue
  if 'PC:' in o:pc[w,ix]=int(re.search(r'PC: (0x[0-9a-f]+)',o)[1],16)
  elif re.match(r'(s_|v_|ds_|buffer_|global_|matrix_)',o):seq[w].append((t,pc.get((w,ix)),o))
result=[]
for w,ops in seq.items():
 ordinal=-1
 for a,b in zip(ops,ops[1:]):
  if not a[2].startswith('v_mmac_'):continue
  ordinal+=1
  if not b[2].startswith('v_mmac_'):continue
  gap=(b[0]-a[0])//period-8
  if gap>=64:result.append(dict(wave=w,after_ordinal=ordinal,cycles=gap,after_pc=hex(a[1]),next_pc=hex(b[1]),next_pc_mod256=b[1]%256,instruction_bytes=8,no_intervening_issued_instructions=True))
stats=(r/'m5out/0/0/stats.txt').read_text();vals={}
for name in ['numBranchTaken','numCyclesWaitIFetchAfterBranchTaken','waitCyclesPerBranchTaken','numCyclesWaveOnCu']:
 m=re.search(r'^system.shaders.CUs00.'+name+r'\s+(\S+)',stats,re.M)
 if m:vals[name]=float(m[1])
out=dict(long_consecutive_mmac_gaps=result,cu0_frontend_counters=vals,interpretation='PC boundary association plus prior PMD fetch probes supports an instruction-fetch hypothesis. This is not a direct fetch-miss trace or proof of a hardware/model defect. Wave-cycle counters must not be summed into elapsed dispatch time.')
(r/'fetch-boundary-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(r.name,vals,[(v['wave'],v['cycles'],v['next_pc_mod256']) for v in result])
