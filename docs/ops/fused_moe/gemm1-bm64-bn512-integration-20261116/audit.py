from pathlib import Path
import json,re,hashlib
root=Path('/home/zhenx/temp/bm64-gemm1-port-20261116')
proof=[]
for role,step,initial,old in [('P0',1,0,(0,1,2)),('C0',2,0,(0,1,2)),('C1',2,1,(1,2,4))]:
 state=initial
 for i in range(100000):
  assert (old[0]&3,old[0]>>2)==(state%3,state//3),(role,i)
  if step==1:old=(old[1],old[2],old[0]^4)
  else:old=(old[2],old[0]^4,old[1]^4)
  state+=step
  if state>=6:state-=6
 proof.append(dict(role=role,steps=100000,slot_and_phase_equal=True))
# This independent audit also checks initialization, separate from parsed waits.
byfreq={}
for freq in ('1800','1500','2200'):
 d=root/('final-gc'+freq)
 if not (d/'metrics.json').exists():continue
 metrics=json.loads((d/'metrics.json').read_text());init={}
 for line in (d/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open():
  if ': s_abarrier_init ' not in line:continue
  m=re.search(r'([0-9a-f]{8}) [0-9a-f]{8}: s_abarrier_init (\d+), (\d+)',line)
  assert m,line
  wave=int(m[1],16);init.setdefault(wave,[]).append((int(m[2]),int(m[3])))
 assert init
 expected=[(i,8 if i in (4,5) else 4) for i in range(16)]
 assert all(v==expected for v in init.values()),init
 waits=metrics['abarrier_wait_ids']
 assert len(waits)==16 and all(ids[0]!=0 and all(0<=i<16 for i in ids) and all(a!=b for a,b in zip(ids,ids[1:])) for ids in waits.values())
 byfreq[freq]=dict(initializing_waves=len(init),initialization_ids_and_counts=expected,all_16_initialized=True,wait_rules_checked_waves=16,first_wait_nonzero=True,no_adjacent_duplicate_wait_id=True,barrier_ids_in_range=True)
# Inspect actual successful emitted assembly, including counter-enabled launches.
artifacts=[]
for d in ('final-gc1800','final-gc1500','final-gc2200','final-equivalence','framework-equivalence'):
 for p in (root/d/'cache').glob('*/*.amdgcn'):
  s=p.read_text()
  counts={x:int(re.search(r'\.'+x+r'_spill_count:\s*(\d+)',s)[1]) for x in ('sgpr','vgpr')}
  scratch=bool(re.search(r'^\s*scratch_\w+',s,re.M))
  private=int(re.search(r'\.private_segment_fixed_size:\s*(\d+)',s)[1])
  assert counts==dict(sgpr=0,vgpr=0) and not scratch and private==0,(p,counts,private)
  artifacts.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),spill_count=counts,scratch_instruction=False,private_segment_fixed_size=private))
report=dict(ticket_equivalence=proof,trace_barrier_rules=byfreq,no_spill_artifacts=artifacts,all_actual_compiled_variants_no_spill=True)
(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('audited',byfreq.keys(),'asm variants',len(artifacts))
