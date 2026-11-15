import re,subprocess,sys,json,bisect,collections
from pathlib import Path
r=Path(sys.argv[1]);a=json.loads((r/'analysis.json').read_text());launch=json.loads((r/'launch.json').read_text());nk=launch['actual_K']//launch['options']['BLOCK_SIZE_K'];nn=launch['actual_N']//launch['options']['BLOCK_SIZE_N']
trace=r/'m5out/0/0/itrace_cu0_se0_xcd0.txt';period=int(re.search(r'GC_CLOCK (\d+)',trace.open().readline())[1])
pattern=r'^ *[0-9]+ 0000000[8c] [0-9a-f]{8}: (s_|v_|ds_|buffer_|global_|matrix_)'
p=subprocess.Popen(['rg',pattern,str(trace)],stdout=subprocess.PIPE,text=True)
rx=re.compile(r'\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: (.*)');events=collections.defaultdict(list)
for line in p.stdout:
 m=rx.match(line)
 if m:events[int(m[2],16)&65535].append((int(m[1])/period,m[3]))
assert p.wait()==0
out={}
for w,ops in events.items():
 waits=[t for t,op in ops if op.startswith('s_abarrier_try_wait')];ids=a['wait_id_audit'][str(w)]['ids'];assert len(waits)==len(ids)
 wait_ids=dict(zip(waits,ids));mm=[t for t,op in ops if op.startswith('v_mmac_f32_16x16x32_fp8_fp8')];assert len(mm)%128==0
 starts=[];ends=[]
 for o in range(len(mm)//128):
  m,n=divmod(o,nn);idx=m*(nn+1)*nk+(0 if n==0 else(n+1)*nk);starts.append(waits[idx]);ends.append(mm[(o+1)*128-1]+8)
 attrs=collections.defaultdict(float)
 for (t,op),(nt,_) in zip(ops,ops[1:]):
  i=bisect.bisect_right(starts,t)-1
  if i<0 or t>=ends[i]:continue
  delta=max(0,min(nt,ends[i])-t)
  if op.startswith('s_abarrier_try_wait'):kind='A_READY' if wait_ids[t] in (1,2) else 'B_READY'
  elif op.startswith('s_waitcnt') and 'lgkmcnt' in op:kind='LDS_waitcnt'
  elif op.startswith('s_waitcnt') and 'vmcnt' in op:kind='VMEM_waitcnt'
  elif op.startswith('v_mmac_f32_16x16x32_fp8_fp8'):
   attrs['MMAC_following_interval_le8']+=min(8,delta);attrs['MMAC_issue_gap']+=max(0,delta-8);continue
  elif op.startswith('s_ebarrier'):kind='partition_ebarrier'
  else:kind='other_instructions_and_issue_gaps'
  attrs[kind]+=delta
 assert abs(sum(attrs.values())-sum(e-b for b,e in zip(starts,ends)))<1e-6,(attrs,sum(e-b for b,e in zip(starts,ends)))
 out[w]=dict(k_window_cycles=sum(e-b for b,e in zip(starts,ends)),attribution_cycles=attrs,definition='per-wave elapsed intervals within same logical K windows as analyze.py; instruction overlap means this is NOT hardware busy accounting and is NOT additive across waves')
(r/'k-wait-attribution.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
