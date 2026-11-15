import sys,re,json,collections
from pathlib import Path
r=Path(sys.argv[1]);rx=re.compile(r"\s*(\d+) ([0-9a-f]{8}) [0-9a-f]{8}: ((?:s_|v_|ds_|buffer_|global_|matrix_).*)")
state=collections.defaultdict(int);windows=collections.defaultdict(list)
for line in (r/'m5out/0/0/itrace_cu0_se0_xcd0.txt').open():
 m=rx.match(line)
 if not m:continue
 w=int(m[2],16);op=m[3]
 if (w&65535)>=4:continue
 if op.startswith('s_setprio '):
  priority=int(op.split()[1],0)
  if priority==2:
   assert state[w]==0;windows[w].append({'start_tick':int(m[1]),'mls':0,'other_global_memory':0})
  elif priority==0:
   assert state[w]==2;windows[w][-1]['end_tick']=int(m[1])
  else:raise AssertionError(priority)
  state[w]=priority
 elif state[w]==2:
  if op.startswith('matrix_load_'):windows[w][-1]['mls']+=1
  if op.startswith(('buffer_load_','global_load_','buffer_store_','global_store_')):windows[w][-1]['other_global_memory']+=1
assert len(windows)==4
for w,ws in windows.items():
 assert state[w]==0 and len(ws)==8,(w,state[w],ws)
 assert all(x['mls']==2 and x['other_global_memory']==0 and x['end_tick']>x['start_tick'] for x in ws),(w,ws)
(r/'a-chunk-issue-priority-audit.json').write_text(json.dumps({f'{w:08x}':ws for w,ws in windows.items()},indent=2)+'\n');print('4 P0 waves: eight A windows each, priority2 for 2 MLS/window, metadata/store at priority0')
